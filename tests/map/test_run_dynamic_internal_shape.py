from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map._load import load_outputs
from pipefunc.typing import Array  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path


def test_dynamic_internal_shape(tmp_path: Path) -> None:
    @pipefunc(output_name="n")
    def f() -> int:
        return 4

    @pipefunc(output_name="x", internal_shape=("n",))
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
            "Expression `'a' + 'b'` must evaluate to an integer but it evaluated to `ab`.",
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


def test_2d_internal_shape() -> None:
    @pipefunc(output_name="n")
    def f(a) -> int:
        return 4 + a

    @pipefunc(output_name="x", internal_shape=("n",))
    def g(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def h(x: int) -> int:
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
    results = pipeline.map({"a": [0, 0]}, run_folder=None, parallel=False)
    assert results["y"].output.tolist() == [[0, 0], [2, 2], [4, 4], [6, 6]]


def test_internal_shape_2nd_step() -> None:
    @pipefunc(output_name="x", internal_shape=("len(x)",))
    def g() -> list[int]:
        n = random.randint(1, 10)  # noqa: S311
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def h(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([g, h])
    pipeline.map({}, run_folder=None, parallel=False)
