"""Tests for `pipefunc.Pipeline` that call `pipeline.update_*`."""

from __future__ import annotations

import numpy as np
import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc


def test_renaming_output_name() -> None:
    @pipefunc(output_name=("c", "d"), renames={"a": "a1"})
    def f(a, b):
        return a + b, 1

    f.update_renames({"c": "c1"}, update_from="current")
    assert f.output_name == ("c1", "d")
    pipeline = Pipeline([f])
    pipeline.update_renames({"c1": "c2"}, update_from="current")
    pipeline.update_renames({"d": "d1"}, overwrite=True, update_from="current")
    assert pipeline["c"].output_name == ("c", "d1")
    pipeline.update_renames({"c": "c1"}, overwrite=True, update_from="original")
    assert pipeline["c1"].output_name == ("c1", "d")
    f2 = pipeline["c1"].copy()
    assert f2.output_name == ("c1", "d")
    f2.update_renames({"c1": "c2"}, update_from="current")
    assert f2.output_name == ("c2", "d")
    f2.update_renames({}, overwrite=True)
    assert f2.output_name == ("c", "d")


def test_update_pipeline_defaults() -> None:
    @pipefunc(output_name="c", defaults={"b": 1}, renames={"a": "a1"})
    def f(a=42, b=69):
        return a + b

    pipeline = Pipeline([f])
    fp = pipeline.functions[0]

    # Test initial parameters and defaults
    assert fp.parameters == ("a1", "b")
    assert fp.defaults == {"a1": 42, "b": 1}

    # Update defaults
    pipeline.update_defaults({"b": 2})
    assert fp.defaults == {"a1": 42, "b": 2}

    # Call function with updated defaults
    assert pipeline(a1=3) == 5

    # Overwrite defaults
    pipeline.update_defaults({"a1": 1, "b": 3}, overwrite=True)
    assert fp.defaults == {"a1": 1, "b": 3}
    assert fp.parameters == ("a1", "b")

    # Call function with new defaults
    assert fp(a1=2) == 5
    assert fp() == 4
    assert fp(a1=2, b=3) == 5

    with pytest.raises(ValueError, match="Unused keyword arguments"):
        pipeline.update_defaults({"does_not_exist": 1})


def test_update_defaults_and_renames_with_pipeline() -> None:
    @pipefunc(output_name="x", defaults={"b": 1}, renames={"a": "a1"})
    def f(a=42, b=69):
        return a + b

    @pipefunc(output_name="y", defaults={"c": 2}, renames={"d": "d1"})
    def g(c=999, d=666):
        return c * d

    # Test initial pipeline parameters and defaults
    assert f.parameters == ("a1", "b")
    assert f.defaults == {"a1": 42, "b": 1}
    assert g.parameters == ("c", "d1")
    assert g.defaults == {"c": 2, "d1": 666}

    # Update defaults and renames within pipeline
    f.update_defaults({"b": 3})
    f.update_renames({"a": "a2"}, update_from="original")
    g.update_defaults({"c": 4})
    g.update_renames({"d": "d2"}, update_from="original")

    # Test updated pipeline parameters and defaults
    assert f.parameters == ("a2", "b")
    assert f.defaults == {"a2": 42, "b": 3}
    assert g.parameters == ("c", "d2")
    assert g.defaults == {"c": 4, "d2": 666}

    # Call functions within pipeline with updated defaults and renames
    pipeline = Pipeline([f, g])
    assert pipeline("x", a2=3) == 6
    assert pipeline("y", c=2, d2=3) == 6
    assert pipeline("y") == 4 * 666


def test_update_renames_pipeline() -> None:
    @pipefunc(output_name="c", renames={"a": "a1"})
    def f(a, b):
        return a, b

    pipeline = Pipeline([f])
    assert pipeline("c", a1="a1", b="b") == ("a1", "b")

    pipeline.update_renames({"a1": "a2"}, update_from="current")
    assert pipeline("c", a2="a2", b="b") == ("a2", "b")

    pipeline.update_renames({"a2": "a3"}, update_from="current")
    assert pipeline("c", a3="a3", b="b") == ("a3", "b")

    pipeline.update_renames({"b": "b1"}, update_from="current")
    assert pipeline("c", a3="a3", b1="b1") == ("a3", "b1")

    with pytest.raises(
        ValueError,
        match="Unused keyword arguments: `a3`. These are not settable renames",
    ):
        pipeline.update_renames({"a3": "foo"}, update_from="original")

    with pytest.raises(
        ValueError,
        match="Unused keyword arguments: `a`. These are not settable renames",
    ):
        pipeline.update_renames({"a": "foo"}, update_from="current")

    pipeline.update_renames({"a": "a5"}, update_from="original")
    assert pipeline("c", a5="a5", b1="b1") == ("a5", "b1")

    pipeline.update_renames({"a": "a6"}, update_from="original", overwrite=True)
    assert pipeline("c", a6="a6", b="b") == ("a6", "b")


def test_pipeline_scope() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    pipeline = Pipeline([f])
    scope = "x"
    pipeline.update_scope(scope, "*")

    assert pipeline(x={"a": 1, "b": 1}) == 2
    assert pipeline(**{"x.a": 1, "x.b": 1}) == 2
    assert pipeline(**{"x.b": 1, "x": {"a": 1}}) == 2


def test_pipeline_scope_no_selected_exception() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    pipeline = Pipeline([f])
    with pytest.raises(ValueError, match="No function's scope was updated"):
        pipeline.update_scope("myscope")


def test_pipeline_scope_partial() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c

    pipeline = Pipeline([f, g])
    scope = "x"
    pipeline.update_scope(scope, inputs="*", outputs={"c"})
    assert pipeline["x.c"].output_name == "x.c"
    assert pipeline["x.c"].parameters == ("x.a", "x.b")
    assert pipeline["d"].parameters == ("x.c",)
    assert pipeline["d"].output_name == "d"
    assert pipeline("d", x={"a": 1, "b": 1}) == 2
    assert pipeline(x={"a": 1, "b": 1}) == 2
    assert pipeline("x.c", x={"a": 1, "b": 1}) == 2
    assert pipeline(**{"x.a": 1, "x.b": 1}) == 2


def test_set_pipeline_scope_on_init() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    pipeline = Pipeline([f], scope="x")
    assert pipeline("x.c", x={"a": 1, "b": 1}) == 2
    assert pipeline(x={"a": 1, "b": 1}) == 2
    assert pipeline("x.c", x={"a": 1, "b": 1}) == 2
    assert pipeline(**{"x.a": 1, "x.b": 1}) == 2
    with pytest.raises(ValueError, match="The provided `scope='a'` cannot be identical "):
        pipeline.update_scope("a", "*")
    pipeline.update_scope("foo", outputs="*")
    pipeline.update_scope("foo", outputs="*")  # twice should be fine
    assert pipeline("foo.c", x={"a": 1, "b": 1}) == 2
    pipeline.update_scope(None, {"x.b"})
    assert pipeline("foo.c", x={"a": 1}, b=1) == 2
    pipeline.update_scope(None, "*")
    assert pipeline("foo.c", a=1, b=1) == 2

    with pytest.raises(ValueError, match="The `renames` should contain"):
        pipeline.update_renames({"a": "qq#.a"})


def test_scope_and_parameter_identical() -> None:
    @pipefunc(output_name="c")
    def f(x, bar):
        return x + bar

    f.update_scope("foo", {"x"}, {"c"})
    assert f.parameters == ("foo.x", "bar")
    assert f.renames == {"x": "foo.x", "c": "foo.c"}

    pipeline1 = Pipeline([f])

    @pipefunc(output_name="d", scope="bar")
    def g(foo):
        return foo

    pipeline2 = Pipeline([g])

    with pytest.raises(ValueError, match="`bar` are used as both parameter and scope"):
        pipeline1 | pipeline2


def test_scope_with_mapspec() -> None:
    @pipefunc(output_name="c", mapspec="a[i] -> c[i]")
    def f(a, b):
        return a + b

    f.update_scope("foo", {"a"}, {"c"})
    assert str(f.mapspec) == "foo.a[i] -> foo.c[i]"
    pipeline = Pipeline([f])
    assert pipeline.mapspecs_as_strings == ["foo.a[i] -> foo.c[i]"]
    results = pipeline.map({"foo": {"a": [0, 1, 2]}, "b": 1}, parallel=False, storage="dict")
    assert results["foo.c"].output.tolist() == [1, 2, 3]


def test_update_scope_output_only() -> None:
    @pipefunc(output_name="z")
    def add(x: int, y: int) -> int:
        assert isinstance(x, int)
        assert isinstance(y, int)
        return x + y

    @pipefunc(output_name="prod")
    def take_sum(z: np.ndarray) -> int:
        return np.prod(z)

    pipeline = Pipeline([(add, "x[i], y[j] -> z[i, j]"), take_sum])
    pipeline.update_scope("foo", outputs={"z"})
    assert pipeline["foo.z"].parameters == ("x", "y")
    assert pipeline["foo.z"].output_name == "foo.z"
    assert pipeline["prod"].parameters == ("foo.z",)
    assert pipeline["prod"].output_name == "prod"


def test_update_scope_from_faq() -> None:
    @pipefunc(output_name="y", scope="foo")
    def f(a, b):
        return a + b

    assert f.renames == {"a": "foo.a", "b": "foo.b", "y": "foo.y"}

    def g(a, b, y):
        return a * b + y

    g_func = PipeFunc(g, output_name="z", renames={"y": "foo.y"})
    assert g_func.parameters == ("a", "b", "foo.y")
    assert g_func.output_name == "z"

    g_func.update_scope("bar", inputs={"a"}, outputs="*")
    assert g_func.parameters == ("bar.a", "b", "foo.y")
    assert g_func.output_name == "bar.z"

    pipeline = Pipeline([f, g_func])
    # all outputs except foo.y, so only bar.z, which becomes baz.z
    pipeline.update_scope("baz", inputs=None, outputs="*", exclude={"foo.y"})
    kwargs = {"foo.a": 1, "foo.b": 2, "bar.a": 3, "b": 4}
    assert pipeline(**kwargs) == 15
    results = pipeline.map(inputs=kwargs, parallel=False, storage="dict")
    assert results["baz.z"].output == 15
    assert pipeline(foo={"a": 1, "b": 2}, bar={"a": 3}, b=4) == 15
