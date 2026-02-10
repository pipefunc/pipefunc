from __future__ import annotations

from typing import TYPE_CHECKING

from pipefunc import Pipeline, pipefunc

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
