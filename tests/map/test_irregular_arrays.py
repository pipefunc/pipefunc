import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.typing import Array


@pytest.fixture
def pipeline() -> Pipeline:
    @pipefunc(output_name="x")
    def generate_ints(n: int) -> list[int]:
        """Generate a list of integers from 0 to n-1."""
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        """Double the input integer."""
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: Array[int]) -> int:
        """Sum a list of integers."""
        return sum(y)

    pipeline = Pipeline([generate_ints, double_it, take_sum])
    pipeline.add_mapspec_axis("n", axis="i*")
    return pipeline


def test_irregular(pipeline: Pipeline) -> None:
    r = pipeline.map(
        inputs={"n": [1, 2, 3, 4, 5]},
        show_progress="rich",
        internal_shapes={"x": (5,)},
    )
    assert r["x"] == []
