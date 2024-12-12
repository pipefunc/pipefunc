from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map._load import load_outputs
from pipefunc.map._shapes import shape_is_resolved
from pipefunc.map._storage_array._base import StorageBase
from pipefunc.typing import Array  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path


def test_dynamic_internal_shape(tmp_path: Path) -> None:
    @pipefunc(output_name="n")
    def f() -> int:
        return 4

    @pipefunc(output_name="x", internal_shape=("?",))
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


def test_2d_internal_shape(tmp_path: Path) -> None:
    counters = {"f": 0, "g": 0, "h": 0}

    @pipefunc(output_name="n")
    def f(a) -> int:
        counters["f"] += 1
        return 4 + a

    @pipefunc(output_name="x", internal_shape=("?",))
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
    _ = pipeline.map({"a": [0, 0]}, run_folder=tmp_path, parallel=False, cleanup=False)
    assert before == counters


def test_internal_shape_2nd_step() -> None:
    @pipefunc(output_name="x", internal_shape=("?",))
    def g() -> list[int]:
        n = random.randint(1, 10)  # noqa: S311
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def h(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([g, h])
    pipeline.map({}, run_folder=None, parallel=False)


def test_internal_shape_2nd_step2() -> None:
    @pipefunc(output_name="x", internal_shape=("?",))
    def g() -> list[int]:
        n = random.randint(1, 10)  # noqa: S311
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def h(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([g, h])
    pipeline.map({}, run_folder=None, parallel=False)


def test_first_returns_2d() -> None:
    @pipefunc(output_name="x", internal_shape=("?", "?"))
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
