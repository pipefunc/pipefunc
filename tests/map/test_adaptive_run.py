import adaptive

from pipefunc import Pipeline, pipefunc
from pipefunc._adaptive import to_adaptive_learner


def test_adaptive_run():
    @pipefunc(output_name="c", cache=True)
    def f_c(a, b):
        return a + b

    @pipefunc(output_name="d", cache=True)
    def f_d(b, c, x=1):  # noqa: ARG001
        return b * c

    @pipefunc(output_name="e", cache=True)
    def f_e(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f_c, f_d, f_e], cache_type="hybrid")

    learner_1d = to_adaptive_learner(pipeline, "e", {"b": 0}, {"a": (0, 10)})
    adaptive.runner.simple(learner_1d, npoints_goal=10)
    assert len(learner_1d.data) == 10

    learner_2d = to_adaptive_learner(pipeline, "e", {}, {"a": (0, 10), "b": (0, 10)})
    adaptive.runner.simple(learner_2d, npoints_goal=10)
    assert len(learner_2d.data) == 10

    learner_3d = to_adaptive_learner(pipeline, "e", {}, {"a": (0, 10), "b": (0, 10), "x": (0, 10)})
    adaptive.runner.simple(learner_3d, npoints_goal=10)
    assert len(learner_3d.data) == 10

    assert len(pipeline.cache.cache) > 10
