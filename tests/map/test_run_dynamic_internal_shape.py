from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map._load import load_outputs
from pipefunc.typing import Array  # noqa: TCH001

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
        ValueError,
        match="Expression `'a' + 'b'` must evaluate to an integer but it evaluated to `ab`.",
    ):
        pipeline.map({"n": 4}, run_folder=None, parallel=False)
