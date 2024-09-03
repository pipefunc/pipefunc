import adaptive
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc._adaptive import to_adaptive_learner


@pytest.mark.parametrize("full_output", [False, True])
def test_adaptive_run(full_output: bool):  # noqa: FBT001
    @pipefunc(output_name="c", cache=True)
    def f_c(a, b):
        return a + b

    @pipefunc(output_name="d", cache=True)
    def f_d(b, c, x=1):
        return b * c

    @pipefunc(output_name="e", cache=True)
    def f_e(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f_c, f_d, f_e], cache_type="hybrid")

    learner_1d = to_adaptive_learner(
        pipeline,
        "e",
        {"b": 0},
        {"a": (0, 10)},
        full_output=full_output,
    )
    adaptive.runner.simple(learner_1d, npoints_goal=10)
    assert len(learner_1d.data) == 10

    learner_2d = to_adaptive_learner(
        pipeline,
        "e",
        {},
        {"a": (0, 10), "b": (0, 10)},
        full_output=full_output,
    )
    adaptive.runner.simple(learner_2d, npoints_goal=10)
    assert len(learner_2d.data) == 10

    learner_3d = to_adaptive_learner(
        pipeline,
        "e",
        {},
        {"a": (0, 10), "b": (0, 10), "x": (0, 10)},
        full_output=full_output,
    )
    adaptive.runner.simple(learner_3d, npoints_goal=10)
    assert len(learner_3d.data) == 10

    assert pipeline.cache is not None
    assert len(pipeline.cache.cache) > 10

    if full_output:
        assert learner_1d.extra_data
        assert learner_2d.extra_data
        assert learner_3d.extra_data


def test_different_output_and_adaptive_name():
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return {"random_object": [c, c + 1, c + 2]}

    pipeline = Pipeline([f, g])

    learner_2d = to_adaptive_learner(
        pipeline,
        output_name="d",
        adaptive_output="c",
        kwargs={},
        adaptive_dimensions={"a": (0, 10), "b": (0, 10)},
        full_output=True,
    )
    adaptive.runner.simple(learner_2d, npoints_goal=10)
    assert len(learner_2d.data) == 10
    assert len(learner_2d.extra_data) == 10

    with pytest.raises(
        ValueError,
        match="If `adaptive_output != output_name`, `full_output` must be True",
    ):
        to_adaptive_learner(
            pipeline,
            output_name="d",
            adaptive_output="c",
            kwargs={},
            adaptive_dimensions={"a": (0, 10), "b": (0, 10)},
            full_output=False,  # This should raise an error
        )
