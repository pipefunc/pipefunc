"""Tests for `pipefunc.Pipeline` with `lazy=True`."""

from __future__ import annotations

import pytest

from pipefunc import Pipeline, pipefunc


def test_pipeline_and_all_arg_combinations_lazy() -> None:
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f2(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f1, f2, f3], debug=True, profile=True, lazy=True)

    fc = pipeline.func("c")
    fd = pipeline.func("d")
    c = f1(a=2, b=3)
    assert fc(a=2, b=3).evaluate() == c == fc(b=3, a=2).evaluate() == 5
    assert fd(a=2, b=3).evaluate() == f2(b=3, c=c) == fd(b=3, c=c).evaluate() == 15

    fe = pipeline.func("e")
    assert (
        fe(a=2, b=3, x=1).evaluate()
        == fe(a=2, b=3, d=15, x=1).evaluate()
        == f3(c=c, d=15, x=1)
        == 75
    )

    all_args = pipeline.all_arg_combinations

    kw = {"a": 2, "b": 3, "x": 1}
    kw["c"] = f1(a=kw["a"], b=kw["b"])
    kw["d"] = f2(b=kw["b"], c=kw["c"])
    kw["e"] = f3(c=kw["c"], d=kw["d"], x=kw["x"])
    for params in all_args["e"]:
        _kw = {k: kw[k] for k in params}
        assert fe(**_kw).evaluate() == kw["e"]


def test_complex_pipeline() -> None:
    def f1(a, b, c, d):
        return a + b + c + d

    def f2(a, b, e):
        return a + b + e

    def f3(a, b, f1):
        return a + b + f1

    def f4(f1, f2, f3):
        return f1 + f2 + f3

    def f5(f1, f4):
        return f1 + f4

    def f6(b, f5):
        return b + f5

    def f7(a, f2, f6):
        return a + f2 + f6

    pipeline = Pipeline([f1, f2, f3, f4, f5, f6, f7], lazy=True)  # type: ignore[list-item]

    r = pipeline("f7", a=1, b=2, c=3, d=4, e=5)
    assert r.evaluate() == 52


def test_lazy_pipeline() -> None:
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f2(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f1, f2, f3], lazy=True)

    f = pipeline.func("e")
    r = f(a=1, b=2, x=3).evaluate()
    assert r == 162
    r = f.call_full_output(a=1, b=2, x=3)["e"].evaluate()
    assert r == 162


def test_hybrid_cache_lazy_warning() -> None:
    @pipefunc(output_name="c", cache=True)
    def f(a, b):
        return a, b

    with pytest.warns(UserWarning, match="Hybrid cache uses function evaluation"):
        Pipeline([f], cache_type="hybrid", lazy=True)
