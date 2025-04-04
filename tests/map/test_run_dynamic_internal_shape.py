from __future__ import annotations

import importlib.util
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map import load_xarray_dataset
from pipefunc.map._load import load_outputs
from pipefunc.map._run_info import RunInfo
from pipefunc.map._shapes import shape_is_resolved
from pipefunc.map._storage_array._base import StorageBase
from pipefunc.typing import Array  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path

has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None
has_xarray = importlib.util.find_spec("xarray") is not None


def _pipeline(dim: Literal["?"] | None) -> Pipeline:
    @pipefunc(output_name="n")
    def f() -> int:
        return 4

    @pipefunc(output_name="x", internal_shape=dim)
    def g(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def h(x: int) -> int:
        assert isinstance(x, int)
        return 2 * x

    @pipefunc(output_name="sum")
    def i(y: Array[int]) -> int:
        return sum(y)

    return Pipeline([f, g, h, i])


@pytest.mark.parametrize("return_results", [True, False])
# reaches the run_folder is None in RunInfo.dump:
@pytest.mark.parametrize("use_run_folder", [True, False])
@pytest.mark.parametrize("dim", ["?", None])
def test_dynamic_internal_shape(
    tmp_path: Path,
    dim: Literal["?"] | None,
    return_results: bool,  # noqa: FBT001
    use_run_folder: bool,  # noqa: FBT001
) -> None:
    pipeline = _pipeline(dim)
    assert pipeline.mapspecs_as_strings == ["... -> x[i]", "x[i] -> y[i]"]
    results = pipeline.map(
        {},
        run_folder=tmp_path if use_run_folder else None,
        return_results=return_results,
        parallel=False,
        storage="dict",
    )
    expected_sum = 12
    expected_y = [0, 2, 4, 6]
    if return_results:
        assert results["sum"].output == expected_sum
        assert results["sum"].output_name == "sum"
    if use_run_folder:
        assert load_outputs("sum", run_folder=tmp_path) == expected_sum
        assert load_outputs("y", run_folder=tmp_path).tolist() == expected_y
        if has_xarray:
            load_xarray_dataset("x", run_folder=tmp_path)
            load_xarray_dataset("y", run_folder=tmp_path)


@pytest.mark.parametrize("return_results", [True, False])
@pytest.mark.parametrize("dim", ["?", None])
@pytest.mark.asyncio
async def test_dynamic_internal_shape_async(
    tmp_path: Path,
    dim: Literal["?"] | None,
    return_results: bool,  # noqa: FBT001
) -> None:
    pipeline = _pipeline(dim)
    assert pipeline.mapspecs_as_strings == ["... -> x[i]", "x[i] -> y[i]"]
    runner = pipeline.map_async(
        {},
        run_folder=tmp_path,
        return_results=return_results,
        executor=ThreadPoolExecutor(),
        storage="dict",
    )
    results = await runner.task
    expected_sum = 12
    expected_y = [0, 2, 4, 6]
    if return_results:
        assert results["sum"].output == expected_sum
        assert results["sum"].output_name == "sum"
    assert load_outputs("sum", run_folder=tmp_path) == expected_sum
    assert load_outputs("y", run_folder=tmp_path).tolist() == expected_y
    if has_xarray:
        load_xarray_dataset("x", run_folder=tmp_path)
        load_xarray_dataset("y", run_folder=tmp_path)


def test_exception(tmp_path: Path) -> None:
    @pipefunc(
        output_name="x",
        # doesn't evaluate to an int
        internal_shape=("'a' + 'b'",),  # type: ignore[arg-type]
    )
    def g(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def h(x: int) -> int:
        assert isinstance(x, int)
        return 2 * x

    pipeline = Pipeline([g, h])
    with pytest.raises(
        TypeError,
        match=re.escape("Internal shape for 'x' must be a tuple of integers or '?'."),
    ):
        pipeline.map({"n": 4}, run_folder=tmp_path, parallel=False, storage="dict")


@pytest.mark.parametrize("return_results", [True, False])
def test_2d_internal_shape_non_dynamic(
    tmp_path: Path,
    return_results: bool,  # noqa: FBT001
) -> None:
    @pipefunc(output_name="n", mapspec="a[j] -> n[j]")
    def f(a) -> int:
        return 4 + a

    @pipefunc(output_name="x", internal_shape=(4,), mapspec="n[j] -> x[i, j]")
    def g(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i, j] -> y[i, j]")
    def h(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([f, g, h], validate_type_annotations=False)
    assert pipeline.mapspecs_as_strings == [
        "a[j] -> n[j]",
        "n[j] -> x[i, j]",
        "x[i, j] -> y[i, j]",
    ]
    expected_y = [[0, 0], [2, 2], [4, 4], [6, 6]]
    results = pipeline.map(
        {"a": [0, 0]},
        run_folder=tmp_path,
        return_results=return_results,
        parallel=False,
        storage="dict",
    )
    if return_results:
        assert results["y"].output.tolist() == expected_y
    assert load_outputs("y", run_folder=tmp_path).tolist() == expected_y


@pytest.mark.skipif(not has_ipywidgets, reason="ipywidgets not installed")
@pytest.mark.parametrize("dim", ["?", None])
@pytest.mark.parametrize("return_results", [True, False])
@pytest.mark.parametrize("scheduling_strategy", ["generation", "eager"])
def test_2d_internal_shape(
    tmp_path: Path,
    dim: Literal["?"] | None,
    return_results: bool,  # noqa: FBT001
    scheduling_strategy: Literal["generation", "eager"],
) -> None:
    counters = {"f": 0, "g": 0, "h": 0}

    @pipefunc(output_name="n")
    def f(a) -> int:
        counters["f"] += 1
        return 4 + a

    @pipefunc(output_name="x", internal_shape=dim)
    def g(n: int) -> list[int]:
        counters["g"] += 1
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def h(x: int) -> int:
        counters["h"] += 1
        return 2 * x

    pipeline = Pipeline([f, g, h])
    assert pipeline.mapspecs_as_strings == [
        "... -> x[i]",
        "x[i] -> y[i]",
    ]
    pipeline.add_mapspec_axis("a", axis="j")
    assert pipeline.mapspecs_as_strings == [
        "a[j] -> n[j]",
        "n[j] -> x[i, j]",
        "x[i, j] -> y[i, j]",
    ]
    expected_y = [[0, 0], [2, 2], [4, 4], [6, 6]]
    results = pipeline.map(
        {"a": [0, 0]},
        run_folder=tmp_path,
        return_results=return_results,
        scheduling_strategy=scheduling_strategy,
        parallel=False,
        storage="dict",
    )
    if return_results:
        assert results["y"].output.tolist() == expected_y
    assert load_outputs("y", run_folder=tmp_path).tolist() == expected_y
    before = counters.copy()
    # Should use existing results
    _ = pipeline.map(
        {"a": [0, 0]},
        run_folder=tmp_path,
        cleanup=False,
        show_progress=True,
        return_results=return_results,
        scheduling_strategy=scheduling_strategy,
        parallel=False,
        storage="dict",
    )
    assert before == counters


@pytest.mark.skipif(not has_ipywidgets, reason="ipywidgets not installed")
@pytest.mark.parametrize("dim", ["?", None])
def test_internal_shape_2nd_step(tmp_path: Path, dim: Literal["?"] | None) -> None:
    @pipefunc(output_name="x", internal_shape=dim)
    def g() -> list[int]:
        n = random.randint(1, 10)  # noqa: S311
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def h(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([g, h])
    results = pipeline.map(
        {},
        run_folder=tmp_path,
        show_progress=True,
        parallel=False,
        storage="dict",
    )
    # Optionally check that results is a dict if available
    if isinstance(results, dict):
        assert isinstance(results, dict)


@pytest.mark.parametrize("dim", ["?", None])
@pytest.mark.parametrize("return_results", [True, False])
def test_internal_shape_2nd_step2(
    tmp_path: Path,
    dim: Literal["?"] | None,
    return_results: bool,  # noqa: FBT001
) -> None:
    @pipefunc(output_name="x", internal_shape=dim)
    def g() -> list[int]:
        return list(range(7))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def h(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([g, h])
    _ = pipeline.map(
        {},
        run_folder=tmp_path,
        return_results=return_results,
        parallel=False,
        storage="dict",
    )
    run_info = RunInfo.load(tmp_path)
    assert run_info.shapes == {"x": ("?",), "y": ("?",)}
    assert run_info.resolved_shapes == {"x": (7,), "y": (7,)}


@pytest.mark.parametrize("internal_shape", [("?", "?"), None])
@pytest.mark.parametrize("return_results", [True, False])
def test_first_returns_2d(
    internal_shape: tuple | None,
    tmp_path: Path,
    return_results: bool,  # noqa: FBT001
) -> None:
    @pipefunc(output_name="x", internal_shape=internal_shape)
    def g() -> npt.NDArray[np.int_]:
        n = random.randint(1, 10)  # noqa: S311
        m = random.randint(1, 10)  # noqa: S311
        return np.arange(n * m).reshape(n, m)

    @pipefunc(output_name="y", mapspec="x[i, j] -> y[i, j]")
    def h(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([g, h])
    result = pipeline.map(
        {},
        run_folder=tmp_path,
        return_results=return_results,
        parallel=False,
        storage="dict",
    )
    expected_y = (2 * load_outputs("x", run_folder=tmp_path)).tolist()
    if return_results:
        assert result["y"].output.tolist() == (2 * result["x"].output).tolist()
        assert isinstance(result["y"].store, StorageBase)
        assert len(result["y"].store.shape) == 2
        assert shape_is_resolved(result["y"].store.full_shape)
    assert load_outputs("y", run_folder=tmp_path).tolist() == expected_y


@pytest.mark.parametrize("dim", ["?", None])
@pytest.mark.parametrize("return_results", [True, False])
def test_first_returns_2d_but_1d_internal(
    dim: Literal["?"] | None,
    tmp_path: Path,
    return_results: bool,  # noqa: FBT001
) -> None:
    @pipefunc(output_name="x", internal_shape=dim)
    def g() -> npt.NDArray[np.int_]:
        n = 4
        m = random.randint(1, 10)  # noqa: S311
        return np.arange(n * m).reshape(n, m)

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def h(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([g, h])
    result = pipeline.map(
        {},
        run_folder=tmp_path,
        return_results=return_results,
        parallel=False,
        storage="dict",
    )
    if return_results:
        assert np.all(result["y"].output[0] == (2 * result["x"].output[0]))
    expected_y0 = 2 * load_outputs("x", run_folder=tmp_path)[0]
    assert np.all(load_outputs("y", run_folder=tmp_path)[0] == expected_y0)
    if return_results:
        assert isinstance(result["y"].store, StorageBase)
        assert (result["y"].store.shape) == (4,)
        assert shape_is_resolved(result["y"].store.full_shape)


@pytest.mark.parametrize("dim", [3, "?", None])
@pytest.mark.parametrize("order", ["selected[i], out2[i]", "out2[i], selected[i]"])
@pytest.mark.parametrize("return_results", [True, False])
def test_dimension_mismatch_bug_with_autogen_axes(
    tmp_path: Path,
    dim: int | Literal["?"],
    order: str,
    return_results: bool,  # noqa: FBT001
) -> None:
    # Fixes issue in https://github.com/pipefunc/pipefunc/pull/465
    # and afterwards https://github.com/pipefunc/pipefunc/pull/466
    internal_shapes = {"out1": dim, "selected": dim} if dim is not None else None
    jobs = [
        {"out1": 0, "out2": 0},
        {"out1": 0, "out2": 0},
        {"out1": 1, "out2": 1},
    ]

    @pipefunc(output_name=("out1", "out2"))
    def split_dicts(jobs) -> tuple[list[str], list[str]]:
        tuples = [(job["out1"], job["out2"]) for job in jobs]
        out1, out2 = zip(*tuples)
        return list(out1), list(out2)

    @pipefunc("selected")
    def selected(out1) -> list[str]:
        return out1

    @pipefunc("processed", mapspec=f"{order} -> processed[i]")
    def process(selected, out2):
        return f"{selected}, {out2}"

    pipeline = Pipeline([split_dicts, selected, process])
    assert pipeline.mapspecs_as_strings == [
        "... -> out1[i], out2[i]",
        "... -> selected[i]",
        f"{order} -> processed[i]",
    ]
    results = pipeline.map(
        {"jobs": jobs},
        internal_shapes=internal_shapes,  # type: ignore[arg-type]
        run_folder=tmp_path,
        return_results=return_results,
        parallel=False,
        storage="dict",
    )
    expected_processed = ["0, 0", "0, 0", "1, 1"]
    if return_results:
        assert results["processed"].output.tolist() == expected_processed
    assert load_outputs("processed", run_folder=tmp_path).tolist() == expected_processed


def test_dynamic_internal_shape_with_irregular_output(tmp_path: Path) -> None:
    @pipefunc(output_name="x", mapspec="n[k] -> x[i, k]")
    def f(n: int, m: int = 0) -> list[int]:
        return list(range(n + m))

    pipeline = Pipeline([f])

    with pytest.raises(
        ValueError,
        match=re.escape("Output shape (3,) of function 'f' (output 'x') does not match"),
    ):
        pipeline.map(inputs={"n": [2, 3]}, run_folder=tmp_path, parallel=False, storage="dict")
    with pytest.raises(
        ValueError,
        match=re.escape("Output shape (1,) of function 'f' (output 'x') does not match"),
    ):
        pipeline.map(inputs={"n": [2, 1]}, run_folder=tmp_path, parallel=False, storage="dict")


@pytest.mark.parametrize("storage", ["dict", "file_array"])
@pytest.mark.parametrize("return_results", [True, False])
def test_dynamic_internal_shape_with_size_1(
    tmp_path: Path,
    storage: str,
    return_results: bool,  # noqa: FBT001
) -> None:
    @pipefunc(output_name="x", mapspec="n[k] -> x[i, k]")
    def fa(n: int, m: int = 0) -> list[int]:
        return list(range(n + m))

    pipeline = Pipeline([fa])
    expected_x = [[0, 0]]
    r = pipeline.map(
        inputs={"n": [1, 1]},
        run_folder=tmp_path,
        return_results=return_results,
        parallel=False,
        storage=storage,
    )
    if return_results:
        assert r["x"].output.tolist() == expected_x
    assert load_outputs("x", run_folder=tmp_path).tolist() == expected_x


@pytest.mark.parametrize("return_results", [True, False])
@pytest.mark.parametrize("manually_set_internal_shape", [True, False])
def test_dynamic_internal_shape_with_multiple_dynamic_axes(
    tmp_path: Path,
    manually_set_internal_shape: bool,  # noqa: FBT001
    return_results: bool,  # noqa: FBT001
) -> None:
    @pipefunc(output_name="x", mapspec="... -> x[i]")
    def fa(n: int) -> list[int]:
        return [0, 1, 2]

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def fb(x: int) -> int:
        assert x in [0, 1, 2]
        return 2 * x

    @pipefunc(output_name="z", mapspec="... -> z[j]")
    def fc(y) -> list[int]:
        assert y.tolist() == [0, 2, 4]
        return [sum(y), sum(y)]

    pipeline = Pipeline([fa, fb, fc])
    if manually_set_internal_shape:
        pipeline["z"].internal_shape = (2,)
    r = pipeline.map(
        inputs={"n": 4},
        run_folder=tmp_path,
        return_results=return_results,
        parallel=False,
        storage="dict",
    )
    expected_z = [6, 6]
    if return_results:
        assert r["z"].output == expected_z
    assert load_outputs("z", run_folder=tmp_path) == expected_z

    pipeline.add_mapspec_axis("n", axis="k")
    assert pipeline.mapspecs_as_strings == [
        "n[k] -> x[i, k]",
        "x[i, k] -> y[i, k]",
        "y[:, k] -> z[j, k]",
    ]
    r = pipeline.map(
        inputs={"n": [4, 4]},
        run_folder=tmp_path,
        return_results=return_results,
        parallel=False,
        storage="dict",
    )
    expected_z2 = [[6, 6], [6, 6]]
    if return_results:
        assert r["z"].output.tolist() == expected_z2
    assert load_outputs("z", run_folder=tmp_path).tolist() == expected_z2


@pytest.mark.parametrize("return_results", [False])
def test_simple_2d(
    tmp_path: Path,
    return_results: bool,  # noqa: FBT001
) -> None:
    @pipefunc(output_name="y", mapspec="x[:, k] -> y[i, k]")
    def fa(x: np.ndarray[Any, np.dtype[np.int64]]) -> np.ndarray[Any, np.dtype[np.int64]]:
        # The input is 1D slices of the input `x`.
        # This function will reduce the first dim of the input and
        # generate a new axis in its place.
        assert x.shape == (2,)
        y = np.hstack([x, x, x])
        assert y.shape == (6,)
        return y

    @pipefunc(output_name="z", mapspec="y[i, k] -> z[i, k]")
    def fb(y: int) -> int:
        return 3 * y

    pipeline = Pipeline([fa, fb])
    x = np.array([[0, 1, 2], [3, 4, 5]])
    r = pipeline.map(
        inputs={"x": x},
        run_folder=tmp_path,
        return_results=return_results,
        parallel=False,
        storage="dict",
    )
    expected_y = [
        [0, 1, 2],
        [3, 4, 5],
        [0, 1, 2],
        [3, 4, 5],
        [0, 1, 2],
        [3, 4, 5],
    ]
    expected_z = [
        [0, 3, 6],
        [9, 12, 15],
        [0, 3, 6],
        [9, 12, 15],
        [0, 3, 6],
        [9, 12, 15],
    ]
    if return_results:
        assert r["y"].output.tolist() == expected_y
        assert r["z"].output.tolist() == expected_z
    assert load_outputs("y", run_folder=tmp_path).tolist() == expected_y
    assert load_outputs("z", run_folder=tmp_path).tolist() == expected_z


@pytest.mark.parametrize("return_results", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("storage", ["file_array", "dict"])
def test_multiple_outputs_with_dynamic_shape_and_individual_outputs_are_nd_arrays(
    tmp_path: Path,
    return_results: bool,  # noqa: FBT001
    parallel: bool,  # noqa: FBT001
    storage: str,
) -> None:
    @pipefunc(("y1", "y2", "y3"), mapspec="... -> y1[i], y2[i], y3[i]")
    def f(x):
        y1 = np.array([[x, x], [x + 1, x + 1]])
        y2 = np.array([[x + 2, x + 2], [x + 3, x + 3]])
        y3 = np.array([[x + 2], [x + 3]])
        assert y1.shape == (2, 2)
        return y1, y2, y3

    pipeline = Pipeline([f])
    pipeline.add_mapspec_axis("x", axis="j")
    assert pipeline.mapspecs_as_strings == ["x[j] -> y1[i, j], y2[i, j], y3[i, j]"]
    results = pipeline.map(
        {"x": [1]},
        run_folder=tmp_path,
        return_results=return_results,
        parallel=parallel,
        storage=storage,
    )
    if return_results:
        y1 = results["y1"].output
        y2 = results["y2"].output
        y3 = results["y3"].output
        assert y1.shape == (2, 1)
        assert y1[0, 0].tolist() == [1, 1]
        assert y1[1, 0].tolist() == [2, 2]
        assert y2.shape == (2, 1)
        assert y2[0, 0].tolist() == [3, 3]
        assert y2[1, 0].tolist() == [4, 4]
        assert y3.shape == (2, 1)
        assert y3[0, 0].tolist() == [3]
        assert y3[1, 0].tolist() == [4]
    y1_loaded = load_outputs("y1", run_folder=tmp_path)
    y2_loaded = load_outputs("y2", run_folder=tmp_path)
    y3_loaded = load_outputs("y3", run_folder=tmp_path)
    assert y1_loaded.shape == (2, 1)
    assert y1_loaded[0, 0].tolist() == [1, 1]
    assert y1_loaded[1, 0].tolist() == [2, 2]
    assert y2_loaded.shape == (2, 1)
    assert y2_loaded[0, 0].tolist() == [3, 3]
    assert y2_loaded[1, 0].tolist() == [4, 4]
    assert y3_loaded.shape == (2, 1)
    assert y3_loaded[0, 0].tolist() == [3]
    assert y3_loaded[1, 0].tolist() == [4]


@pytest.mark.parametrize("return_results", [True, False])
@pytest.mark.parametrize("storage_id", ["file_array", "shared_memory_dict", "dict"])
@pytest.mark.parametrize("asarray", [True, False])
def test_inhomogeneous_array(
    tmp_path: Path,
    storage_id: str,
    return_results: bool,  # noqa: FBT001
    asarray: bool,  # noqa: FBT001
) -> None:
    internal_shape = (2,)

    @pipefunc(output_name="x", internal_shape=internal_shape, mapspec="... -> x[i]")
    def f():
        if asarray:
            x = np.empty(internal_shape, dtype=object)
            x[0] = ("yo", ("lo",))
            x[1] = ("foo",)
        else:
            x = [("yo", ("lo",)), ("foo",)]
        return x

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def g(x):
        return x

    pipeline = Pipeline([f, g])
    results = pipeline.map(
        {},
        run_folder=tmp_path,
        return_results=return_results,
        parallel=False,
        storage=storage_id,
    )
    if return_results:
        assert results["x"].output[0] == ("yo", ("lo",))
        assert results["x"].output[1] == ("foo",)
        assert results["y"].output[0] == ("yo", ("lo",))
        assert results["y"].output[1] == ("foo",)
    x = load_outputs("x", run_folder=tmp_path)
    y = load_outputs("y", run_folder=tmp_path)
    assert x[0] == ("yo", ("lo",))
    assert x[1] == ("foo",)
    assert y[0] == ("yo", ("lo",))
    assert y[1] == ("foo",)
