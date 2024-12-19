from __future__ import annotations

import importlib.util
import random
import re
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map._load import load_outputs
from pipefunc.map._run_info import RunInfo
from pipefunc.map._shapes import shape_is_resolved
from pipefunc.map._storage_array._base import StorageBase
from pipefunc.typing import Array  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path

has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None


@pytest.mark.parametrize("dim", ["?", None])
def test_dynamic_internal_shape(tmp_path: Path, dim: Literal["?"] | None) -> None:
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

    pipeline = Pipeline([f, g, h, i])
    assert pipeline.mapspecs_as_strings == ["... -> x[i]", "x[i] -> y[i]"]
    results = pipeline.map({}, run_folder=tmp_path, parallel=False)
    assert results["sum"].output == 12
    assert results["sum"].output_name == "sum"
    assert load_outputs("sum", run_folder=tmp_path) == 12


def test_exception():
    @pipefunc(
        output_name="x",
        internal_shape=("'a' + 'b'",),  # doesn't evaluate to an int
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
        match=re.escape(
            "Internal shape for 'x' must be a tuple of integers or '?'.",
        ),
    ):
        pipeline.map({"n": 4}, run_folder=None, parallel=False)


# Same as test_dynamic_internal_shape but specifying the internal shape in @pipefunc
def test_2d_internal_shape_non_dynamic() -> None:
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
    results = pipeline.map({"a": [0, 0]}, run_folder=None, parallel=False)
    assert results["y"].output.tolist() == [[0, 0], [2, 2], [4, 4], [6, 6]]


@pytest.mark.skipif(not has_ipywidgets, reason="ipywidgets not installed")
@pytest.mark.parametrize("dim", ["?", None])
def test_2d_internal_shape(tmp_path: Path, dim: Literal["?"] | None) -> None:
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
    results = pipeline.map({"a": [0, 0]}, run_folder=tmp_path, parallel=False)
    assert results["y"].output.tolist() == [[0, 0], [2, 2], [4, 4], [6, 6]]
    before = counters.copy()
    # Should use existing results
    _ = pipeline.map(
        {"a": [0, 0]},
        run_folder=tmp_path,
        parallel=False,
        cleanup=False,
        show_progress=True,
    )
    assert before == counters


@pytest.mark.skipif(not has_ipywidgets, reason="ipywidgets not installed")
@pytest.mark.parametrize("dim", ["?", None])
def test_internal_shape_2nd_step(dim: Literal["?"] | None) -> None:
    @pipefunc(output_name="x", internal_shape=dim)
    def g() -> list[int]:
        n = random.randint(1, 10)  # noqa: S311
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def h(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([g, h])
    pipeline.map({}, run_folder=None, parallel=False, show_progress=True)


@pytest.mark.parametrize("dim", ["?", None])
def test_internal_shape_2nd_step2(tmp_path: Path, dim: Literal["?"] | None) -> None:
    @pipefunc(output_name="x", internal_shape=dim)
    def g() -> list[int]:
        n = random.randint(1, 10)  # noqa: S311
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def h(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([g, h])
    pipeline.map({}, run_folder=tmp_path, parallel=False)
    run_info = RunInfo.load(tmp_path)
    assert run_info.shapes == {"x": ("?",), "y": ("?",)}


@pytest.mark.parametrize("internal_shape", [("?", "?"), None])
def test_first_returns_2d(internal_shape: tuple | None) -> None:
    @pipefunc(output_name="x", internal_shape=internal_shape)
    def g() -> npt.NDArray[np.int_]:
        n = random.randint(1, 10)  # noqa: S311
        m = random.randint(1, 10)  # noqa: S311
        return np.arange(n * m).reshape(n, m)

    @pipefunc(output_name="y", mapspec="x[i, j] -> y[i, j]")
    def h(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([g, h])
    result = pipeline.map({}, run_folder=None, parallel=False)
    assert result["y"].output.tolist() == (2 * result["x"].output).tolist()
    assert isinstance(result["y"].store, StorageBase)
    assert len(result["y"].store.shape) == 2
    assert shape_is_resolved(result["y"].store.full_shape)


@pytest.mark.parametrize("dim", ["?", None])
def test_first_returns_2d_but_1d_internal(dim: Literal["?"] | None) -> None:
    @pipefunc(output_name="x", internal_shape=dim)
    def g() -> npt.NDArray[np.int_]:
        n = 4
        m = random.randint(1, 10)  # noqa: S311
        return np.arange(n * m).reshape(n, m)

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def h(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([g, h])
    result = pipeline.map({}, run_folder=None, parallel=False)
    assert np.all(result["y"].output[0] == (2 * result["x"].output[0]))
    assert isinstance(result["y"].store, StorageBase)
    assert (result["y"].store.shape) == (4,)
    assert shape_is_resolved(result["y"].store.full_shape)


@pytest.mark.parametrize("dim", [3, "?", None])
@pytest.mark.parametrize("order", ["selected[i], out2[i]", "out2[i], selected[i]"])
def test_dimension_mismatch_bug_with_autogen_axes(
    dim: int | Literal["?"],
    order: str,
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
        parallel=False,
        storage="dict",
    )
    assert results["processed"].output.tolist() == ["0, 0", "0, 0", "1, 1"]
