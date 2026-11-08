from __future__ import annotations

from typing import TYPE_CHECKING

from pipefunc import Pipeline, pipefunc
from pipefunc.typing import Array  # noqa: TC001

if TYPE_CHECKING:
    from pipefunc.map._result import ResultDict


def test_cached_resultdict_reuses_mutable_instance() -> None:
    """Mutating returned results leaks into cached copy reused by Pipeline.run."""

    @pipefunc(output_name="values")
    def identity(x: list[int]) -> list[int]:
        return x

    inner = Pipeline(
        [
            (identity, "x[i] -> values[i]"),
        ],
        cache_type="simple",
    )

    @pipefunc(output_name="results", cache=True)
    def run_inner(x: list[int]) -> ResultDict:
        return inner.map({"x": x}, parallel=False)

    outer = Pipeline([run_inner], cache_type="simple")

    first = outer.run("results", kwargs={"x": [1, 2, 3]})
    assert first["values"].output.tolist() == [1, 2, 3]

    # User mutates the returned ResultDict in place.
    first["values"].output[1] = 999

    second = outer.run("results", kwargs={"x": [1, 2, 3]})

    # Fails on current implementation because the cached ResultDict is reused.
    assert second["values"].output[1] == 2


def test_map_cached_reduction_over_list_outputs() -> None:
    """Cached reductions must accept object arrays whose elements are lists."""

    @pipefunc(output_name="scores", mapspec="x[i] -> scores[i]")
    def elementwise(x: float) -> list[float]:
        return [x, x**2]

    @pipefunc(output_name="worst", cache=True)
    def reduce(scores: Array[list[float]]) -> float:
        return max(max(s) for s in scores)

    pipeline = Pipeline([elementwise, reduce], cache_type="simple")
    result = pipeline.map({"x": [1, 2, 3]}, parallel=False, storage="dict")
    assert result["worst"].output == 9
    assert pipeline.cache is not None
    assert len(pipeline.cache) == 1
