"""Tests for pipefunc.Pipeline."""

from __future__ import annotations

import pickle
import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pipefunc import NestedPipeFunc, PipeFunc, Pipeline, pipefunc
from pipefunc._cache import LRUCache
from pipefunc.exceptions import UnusedParametersError
from pipefunc.resources import Resources

if TYPE_CHECKING:
    from pathlib import Path


def test_pipeline_and_all_arg_combinations() -> None:
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f2(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f1, f2, f3], debug=True, profile=True)

    fc = pipeline.func("c")
    fd = pipeline.func("d")
    c = f1(a=2, b=3)
    assert fc(a=2, b=3) == c == fc(b=3, a=2) == 5
    assert fd(a=2, b=3) == f2(b=3, c=c) == fd(b=3, c=c) == 15

    fe = pipeline.func("e")
    assert fe(a=2, b=3, x=1) == fe(a=2, b=3, d=15, x=1) == f3(c=c, d=15, x=1) == 75

    all_args = pipeline.all_arg_combinations
    assert all_args == {
        "c": {("a", "b")},
        "d": {("a", "b", "x"), ("b", "c", "x")},
        "e": {("a", "b", "d", "x"), ("a", "b", "x"), ("b", "c", "x"), ("c", "d", "x")},
    }
    assert pipeline.all_root_args == {
        "c": ("a", "b"),
        "d": ("a", "b", "x"),
        "e": ("a", "b", "x"),
    }

    kw = {"a": 2, "b": 3, "x": 1}
    kw["c"] = f1(a=kw["a"], b=kw["b"])
    kw["d"] = f2(b=kw["b"], c=kw["c"])
    kw["e"] = f3(c=kw["c"], d=kw["d"], x=kw["x"])
    for params in all_args["e"]:
        _kw = {k: kw[k] for k in params}
        assert fe(**_kw) == kw["e"]

    # Test NestedPipeFunc
    f_nested = NestedPipeFunc([f1, f2])
    assert f_nested.output_name == ("c", "d")
    assert f_nested.parameters == ("a", "b", "x")
    assert f_nested.defaults == {"x": 1}
    assert f_nested(a=2, b=3) == (5, 15)
    assert f_nested.renames == {}
    f_nested.update_renames({"a": "a1", "b": "b1"})
    assert f_nested.renames == {"a": "a1", "b": "b1"}
    f_nested.update_renames({"a": "a2", "b": "b2"}, overwrite=True, update_from="original")
    assert f_nested.renames == {"a": "a2", "b": "b2"}
    assert f_nested.parameters == ("a2", "b2", "x")
    f_nested_copy = f_nested.copy()
    assert f_nested_copy.renames == f_nested.renames
    assert f_nested_copy(a2=2, b2=3) == (5, 15)
    f_nested_copy.update_renames({}, overwrite=True)
    pipeline = Pipeline([f_nested_copy, f3])
    assert pipeline("e", a=2, b=3, x=1) == 75

    assert str(pipeline).startswith("Pipeline:")


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


@pytest.mark.parametrize(
    "f2",
    [
        PipeFunc(
            lambda b, c, x: b * c * x,
            output_name="d",
            renames={"x": "xx"},
        ),
        PipeFunc(lambda b, c, xx: b * c * xx, output_name="d"),
    ],
)
def test_pipeline_and_all_arg_combinations_rename(f2):
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f1, f2, f3], debug=True, profile=True)

    fc = pipeline.func("c")
    fd = pipeline.func("d")
    c = f1(a=2, b=3)
    assert fc(a=2, b=3) == c == fc(b=3, a=2) == 5
    assert fd(a=2, b=3, xx=1) == f2(b=3, c=c, xx=1) == fd(b=3, c=c, xx=1) == 15

    fe = pipeline.func("e")
    assert fe(a=2, b=3, x=1, xx=1) == fe(a=2, b=3, d=15, x=1) == f3(c=c, d=15, x=1) == 75

    all_args = pipeline.all_arg_combinations
    assert all_args == {
        "c": {("a", "b")},
        "d": {("a", "b", "xx"), ("b", "c", "xx")},
        "e": {
            ("a", "b", "d", "x"),
            ("a", "b", "x", "xx"),
            ("b", "c", "x", "xx"),
            ("c", "d", "x"),
        },
    }

    assert pipeline.all_root_args == {
        "c": ("a", "b"),
        "d": ("a", "b", "xx"),
        "e": ("a", "b", "x", "xx"),
    }


def test_disjoint_pipelines() -> None:
    @pipefunc(output_name="x")
    def f(a, b):
        return a + b

    @pipefunc(output_name="y")
    def g(c, d):
        return c * d

    p = Pipeline([f, g])
    assert p("x", a=1, b=2) == 3
    assert p("y", c=3, d=4) == 12


def test_different_defaults() -> None:
    @pipefunc(output_name="c")
    def f(a, b=1):
        return a + b

    @pipefunc(output_name="y")
    def g(c, b=2):
        return c * b

    with pytest.raises(ValueError, match="Inconsistent default values"):
        Pipeline([f, g])


def test_output_name_in_kwargs() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    p = Pipeline([f])
    with pytest.raises(ValueError, match="cannot be provided in"):
        assert p("a", a=1)


def test_profiling() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c, b=2):
        return c * b

    p = Pipeline([f, g], debug=True, profile=True)
    p("d", a=1, b=2)
    p.print_profiling_stats()
    for f in p.functions:
        f.profile = False
    with pytest.raises(ValueError, match="Profiling is not enabled"):
        p.print_profiling_stats()


def test_pipe_func_and_execution() -> None:
    def func1(a, b=2):
        return a + b

    def func2(x):
        return 2 * x

    def func3(y, z=3):
        return y - z

    pipe_func1 = PipeFunc(func1, "out1", renames={"a": "a1"})
    pipe_func2 = PipeFunc(func2, "out2", renames={"x": "x2"})
    pipe_func3 = PipeFunc(func3, "out3", renames={"y": "y3", "z": "z3"})

    pipeline = Pipeline([pipe_func1, pipe_func2, pipe_func3], debug=True, profile=True)

    # Create _PipelineAsFunc instances
    function1 = pipeline.func("out1")
    function2 = pipeline.func("out2")
    function3 = pipeline.func("out3")

    # Test calling the functions with keyword arguments
    assert function1(a1=3, b=2) == 5
    assert function2(x2=4) == 8
    assert function3(y3=9, z3=3) == 6

    # Test calling the functions with dict arguments
    assert function1.call_with_dict({"a1": 3, "b": 2}) == 5
    assert function2.call_with_dict({"x2": 4}) == 8
    assert function3.call_with_dict({"y3": 9, "z3": 3}) == 6

    # Test calling the functions with `execute` method
    assert function1.call_with_root_args(3, 2) == 5
    assert function1.call_with_root_args(a1=3, b=2) == 5
    assert function2.call_with_root_args(4) == 8
    assert function3.call_with_root_args(9, 3) == 6

    # Test the pipeline object itself
    assert pipeline("out1", a1=3, b=2) == 5
    assert pipeline("out2", x2=4) == 8
    assert pipeline("out3", y3=9, z3=3) == 6


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


def test_tuple_outputs() -> None:
    cache = True

    @pipefunc(
        output_name=("c", "_throw"),
        profile=True,
        debug=True,
        cache=cache,
        output_picker=dict.__getitem__,
    )
    def f_c(a, b):
        return {"c": a + b, "_throw": 1}

    @pipefunc(output_name=("d", "e"), cache=cache)
    def f_d(b, c, x=1):  # noqa: ARG001
        return b * c, 1

    @pipefunc(output_name=("g", "h"), output_picker=getattr, cache=cache)
    def f_g(c, e, x=1):  # noqa: ARG001
        from types import SimpleNamespace

        print(f"Called f_g with c={c} and e={e}")
        return SimpleNamespace(g=c + e, h=c - e)

    @pipefunc(output_name="i", cache=cache)
    def f_i(h, g):
        return h + g

    pipeline = Pipeline(
        [f_c, f_d, f_g, f_i],
        debug=True,
        profile=True,
        cache_type="lru",
        lazy=True,
        cache_kwargs={"shared": False},
    )
    f = pipeline.func("i")
    r = f.call_full_output(a=1, b=2, x=3)["i"].evaluate()
    assert r == f(a=1, b=2, x=3).evaluate()
    assert (
        pipeline.root_args("g")
        == pipeline.root_args("h")
        == pipeline.root_args(("g", "h"))
        == ("a", "b", "x")
    )
    key = (("d", "e"), (("a", 1), ("b", 2), ("x", 3)))
    assert pipeline.cache is not None
    assert pipeline.cache.cache[key].evaluate() == (6, 1)
    assert pipeline.func(("g", "h"))(a=1, b=2, x=3).evaluate().g == 4
    assert pipeline.func_dependencies("i") == [("c", "_throw"), ("d", "e"), ("g", "h")]
    assert pipeline.func_dependents("c") == [("d", "e"), ("g", "h"), "i"]

    assert (
        pipeline.func_dependencies("g")
        == pipeline.func_dependencies("h")
        == pipeline.func_dependencies(("g", "h"))
        == [("c", "_throw"), ("d", "e")]
    )

    f = pipeline.func(("g", "h"))
    r = f(a=1, b=2, x=3).evaluate()
    assert r.g == 4
    assert r.h == 2

    edges = {
        (pipeline["c"], pipeline["d"]): {"arg": "c"},
        (pipeline["c"], pipeline["g"]): {"arg": "c"},
        ("a", pipeline["c"]): {"arg": "a"},
        ("b", pipeline["c"]): {"arg": "b"},
        ("b", pipeline["d"]): {"arg": "b"},
        (pipeline["d"], pipeline["g"]): {"arg": "e"},
        ("x", pipeline["d"]): {"arg": "x"},
        ("x", pipeline["g"]): {"arg": "x"},
        (pipeline["g"], pipeline["i"]): {"arg": ("h", "g")},
    }
    assert edges == dict(pipeline.graph.edges)

    assert dict(pipeline.graph.nodes) == {
        pipeline["c"]: {},
        "a": {},
        "b": {},
        pipeline["d"]: {},
        "x": {},
        pipeline["g"]: {},
        pipeline["i"]: {},
    }


@pytest.mark.parametrize("cache", [True, False])
def test_full_output(cache, tmp_path: Path):
    from pipefunc import Pipeline

    @pipefunc(output_name="f1")
    def f1(a, b):
        return a + b

    @pipefunc(output_name=("f2i", "f2j"))
    def f2(f1):
        return 2 * f1, 1

    @pipefunc(output_name="f3")
    def f3(a, f2i):
        return a + f2i

    if cache:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_kwargs = {"cache_type": "disk", "cache_kwargs": {"cache_dir": cache_dir}}
    else:
        cache_kwargs = {}
    pipeline = Pipeline([f1, f2, f3], **cache_kwargs)  # type: ignore[arg-type]
    for f in pipeline.functions:
        f.cache = cache
    pipeline("f3", a=1, b=2)
    func = pipeline.func("f3")
    assert func.call_full_output(a=1, b=2) == {
        "a": 1,
        "b": 2,
        "f1": 3,
        "f2i": 6,
        "f2j": 1,
        "f3": 7,
    }
    if cache:
        assert len(list(cache_dir.glob("*.pkl"))) == 3


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


@pipefunc(output_name="test_function")
def test_function(arg1: str, arg2: str) -> str:
    return f"{arg1} {arg2}"


pipeline = Pipeline([test_function])


def test_function_pickling() -> None:
    # Get the _PipelineAsFunc instance from the pipeline
    func = pipeline.func("test_function")

    # Pickle the _PipelineAsFunc instance
    pickled_func = pickle.dumps(func)

    # Unpickle the _PipelineAsFunc instance
    unpickled_func = pickle.loads(pickled_func)  # noqa: S301

    # Assert that the unpickled instance has the same attributes
    assert unpickled_func.output_name == "test_function"
    assert unpickled_func.root_args == ("arg1", "arg2")

    # Assert that the unpickled instance behaves the same as the original
    result = unpickled_func(arg1="hello", arg2="world")
    assert result == "hello world"

    # Assert that the call_with_root_args method is recreated after unpickling
    assert unpickled_func.call_with_root_args is not None
    assert unpickled_func.call_with_root_args.__signature__.parameters.keys() == {
        "arg1",
        "arg2",
    }


def test_drop_from_pipeline() -> None:
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f2(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f1, f2, f3])
    assert "d" in pipeline
    pipeline.drop(output_name="d")
    assert "d" not in pipeline

    pipeline = Pipeline([f1, f2, f3])
    assert "d" in pipeline
    pipeline.drop(f=pipeline["d"])

    pipeline = Pipeline([f1, f2, f3])

    @pipefunc(output_name="e")
    def f4(c, d, x=1):
        return c * d * x

    pipeline.replace(f4)
    assert len(pipeline.functions) == 3
    assert pipeline["e"].__name__ == "f4"

    pipeline.replace(f3, pipeline["e"])
    assert len(pipeline.functions) == 3
    assert pipeline["e"].__name__ == "f3"

    with pytest.raises(ValueError, match="Either `f` or `output_name` should be provided"):
        pipeline.drop()


def test_used_variable() -> None:
    @pipefunc(output_name="c", cache=True)
    def f1(a, b):
        return a + b

    pipeline = Pipeline([f1])  # automatically sets cache_type="lru" because of cache=True
    assert isinstance(pipeline.cache, LRUCache)
    with pytest.raises(UnusedParametersError, match="Unused keyword arguments"):
        pipeline("c", a=1, b=2, doesnotexist=3)

    pipeline("c", a=1, b=2)

    # Test regression with cache:
    def f(a):
        return a

    pipeline = Pipeline([PipeFunc(f, output_name="c", cache=True)], cache_type="lru")
    ff = pipeline.func("c")
    assert ff(a=1) == 1
    assert ff(a=1) == 1  # should not raise an error


def test_handle_error() -> None:
    @pipefunc(output_name="c")
    def f1(a, b):  # noqa: ARG001
        msg = "Test error"
        raise ValueError(msg)

    pipeline = Pipeline([f1])
    try:
        pipeline("c", a=1, b=2)
    except ValueError as e:
        msg = "Error occurred while executing function `f1(a=1, b=2)`"
        assert msg in str(e) or msg in str(e.__notes__)  # type: ignore[attr-defined]  # noqa: PT017
        # NOTE: with pytest.raises match="..." does not work
        # with add_note for some reason on my Mac, however,
        # on CI it works fine (Linux)...


def test_full_output_cache() -> None:
    ran_f1 = False
    ran_f2 = False

    @pipefunc(output_name="c", cache=True)
    def f1(a, b):
        nonlocal ran_f1
        if ran_f2:
            raise RuntimeError
        ran_f1 = True
        return a + b

    @pipefunc(output_name="d", cache=True)
    def f2(b, c, x=1):
        nonlocal ran_f2
        if ran_f2:
            raise RuntimeError
        ran_f2 = True
        return b * c * x

    pipeline = Pipeline([f1, f2], cache_type="hybrid")
    f = pipeline.func("d")
    r = f.call_full_output(a=1, b=2, x=3)
    expected = {"a": 1, "b": 2, "c": 3, "d": 18, "x": 3}
    assert r == expected
    assert pipeline.cache is not None
    assert len(pipeline.cache) == 2
    r = f.call_full_output(a=1, b=2, x=3)
    assert r == expected
    r = f(a=1, b=2, x=3)
    assert r == 18


def test_output_picker_single_output() -> None:
    @pipefunc(output_name=("y",), output_picker=dict.__getitem__)
    def f(a, b):
        return {"y": a + b, "_throw": 1}

    pipeline = Pipeline([f])
    assert pipeline("y", a=1, b=2) == 3


def test_independent_axes_in_mapspecs_with_disconnected_chains() -> None:
    @pipefunc(output_name=("c", "d"), mapspec="a[i] -> c[i], d[i]")
    def f(a: int, b: int):
        return a + b, 1

    @pipefunc(output_name="z", mapspec="x[i], y[i] -> z[i]")
    def g(x, y):
        return x + y

    pipeline = Pipeline([f, g])
    assert pipeline.mapspecs_as_strings == [
        "a[i] -> c[i], d[i]",
        "x[i], y[i] -> z[i]",
    ]
    assert pipeline.independent_axes_in_mapspecs("c") == {"i"}
    assert pipeline.independent_axes_in_mapspecs("d") == {"i"}
    assert pipeline.independent_axes_in_mapspecs(("c", "d")) == {"i"}
    assert pipeline.independent_axes_in_mapspecs("z") == {"i"}

    pipeline.add_mapspec_axis("b", axis="j")
    assert pipeline.mapspecs_as_strings == [
        "a[i], b[j] -> c[i, j], d[i, j]",
        "x[i], y[i] -> z[i]",
    ]
    assert pipeline.independent_axes_in_mapspecs("c") == {"i", "j"}
    assert pipeline.independent_axes_in_mapspecs("d") == {"i", "j"}
    assert pipeline.independent_axes_in_mapspecs(("c", "d")) == {"i", "j"}
    assert pipeline.independent_axes_in_mapspecs("z") == {"i"}

    pipeline.add_mapspec_axis("x", axis="j")
    pipeline.add_mapspec_axis("y", axis="j")
    assert pipeline.mapspecs_as_strings == [
        "a[i], b[j] -> c[i, j], d[i, j]",
        "x[i, j], y[i, j] -> z[i, j]",
    ]
    assert pipeline.independent_axes_in_mapspecs("c") == {"i", "j"}
    assert pipeline.independent_axes_in_mapspecs("d") == {"i", "j"}
    assert pipeline.independent_axes_in_mapspecs(("c", "d")) == {"i", "j"}
    assert pipeline.independent_axes_in_mapspecs("z") == {"i", "j"}

    with pytest.raises(
        ValueError,
        match="The provided `pipefuncs` should have only one leaf node, not 2.",
    ):
        NestedPipeFunc([f, g])


def test_max_single_execution_per_call() -> None:
    counter = {"f_c": 0, "f_d": 0, "f_e": 0}

    @pipefunc(output_name="c")
    def f_c(a, b):
        assert counter["f_c"] == 0
        counter["f_c"] += 1
        print("c")
        return a + b

    @pipefunc(output_name="d")
    def f_d(b, c, x=1):
        assert counter["f_d"] == 0
        counter["f_d"] += 1
        return b * c * x

    @pipefunc(output_name="e")
    def f_e(c, d, x=1):
        assert counter["f_e"] == 0
        counter["f_e"] += 1
        return c * d * x

    pipeline = Pipeline([f_c, f_d, f_e])
    pipeline("e", a=1, b=2, x=3)
    assert counter == {"f_c": 1, "f_d": 1, "f_e": 1}


def test_setting_defaults() -> None:
    @pipefunc(output_name="c", defaults={"b": 2}, renames={"a": "a1"})
    def f(a, b=1):
        return a + b

    assert f.parameters == ("a1", "b")
    assert f.defaults == {"b": 2}
    with pytest.raises(ValueError, match="Unexpected keyword arguments"):
        f(a=0)

    assert f(a1=0) == 2

    pipeline = Pipeline([f])
    assert pipeline("c", a1=0) == 2
    assert pipeline("c", a1="a1", b="b") == "a1b"

    with pytest.raises(ValueError, match="Unexpected `defaults` arguments"):

        @pipefunc(output_name="b", defaults={"a": 2}, renames={"a": "a1"})
        def g(a):
            return a

    @pipefunc(output_name="c", defaults={"a": "a_new", "b": "b_new"}, renames={"a": "b", "b": "a"})
    def h(a="a", b="b"):
        return a, b

    assert h() == ("b_new", "a_new")
    assert h(a="aa", b="bb") == ("bb", "aa")


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


def test_subpipeline() -> None:
    @pipefunc(output_name=("c", "d"))
    def f(a: int, b: int):
        return a + b, 1

    @pipefunc(output_name="z")
    def g(x, y):
        return x + y

    pipeline = Pipeline([f, g])
    partial = pipeline.subpipeline(inputs=["a", "b"])  # type: ignore[arg-type]
    assert [f.output_name for f in partial.functions] == [("c", "d")]

    partial = pipeline.subpipeline(inputs=["a", "b", "x", "y"])  # type: ignore[arg-type]
    assert [f.output_name for f in partial.functions] == [("c", "d"), "z"]

    partial = pipeline.subpipeline(output_names=[("c", "d")])  # type: ignore[arg-type]
    assert [f.output_name for f in partial.functions] == [("c", "d")]

    with pytest.raises(ValueError, match="Cannot construct a partial pipeline"):
        pipeline.subpipeline(inputs=["a"])  # type: ignore[arg-type]

    @pipefunc(output_name="h")
    def h(c):
        return c

    pipeline = Pipeline([f, g, h])
    partial = pipeline.subpipeline(inputs=["a", "b"])  # type: ignore[arg-type]
    assert [f.output_name for f in partial.functions] == [("c", "d"), "h"]

    partial = pipeline.subpipeline(output_names=["h"])  # type: ignore[arg-type]
    assert partial.topological_generations.root_args == ["a", "b"]
    assert partial.root_nodes == ["a", "b"]

    partial = pipeline.subpipeline(output_names=["h"], inputs=["c"])  # type: ignore[arg-type]
    assert partial.topological_generations.root_args == ["c"]
    assert [f.output_name for f in partial.functions] == ["h"]

    with pytest.raises(
        ValueError,
        match="At least one of `inputs` or `output_names` should be provided",
    ):
        pipeline.subpipeline()


def test_nest_all() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c + 1

    pipeline = Pipeline([f, g])
    assert pipeline("d", a=1, b=2) == 4
    nested = pipeline.nest_funcs("*")
    assert nested.output_name == ("c", "d")
    assert nested(a=1, b=2) == (3, 4)
    assert pipeline("d", a=1, b=2) == 4
    assert len(pipeline.functions) == 1


def test_missing_kw() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    pipeline = Pipeline([f])
    with pytest.raises(
        ValueError,
        match=re.escape("Missing value for argument `b` in `f(...) → c`."),
    ):
        pipeline("c", a=1)


def test_join_pipelines() -> None:
    def f(a, b):
        return a + b

    def g(a, b):
        return a * b

    pipeline1 = Pipeline([PipeFunc(f, "f")], debug=True)
    pipeline2 = Pipeline([PipeFunc(g, "g")], debug=False)
    pipeline = pipeline1.join(pipeline2)
    assert pipeline("f", a=1, b=2) == 3
    assert pipeline("g", a=1, b=2) == 2
    assert pipeline.debug

    pipeline = pipeline1 | pipeline2
    assert pipeline("f", a=1, b=2) == 3
    assert pipeline("g", a=1, b=2) == 2
    assert pipeline.debug

    pipeline = pipeline1 | PipeFunc(g, "g")
    assert pipeline("f", a=1, b=2) == 3
    assert pipeline("g", a=1, b=2) == 2
    assert pipeline.debug

    with pytest.raises(
        TypeError,
        match="Only `Pipeline` or `PipeFunc` instances can be joined",
    ):
        pipeline1 | g  # type: ignore[operator]


def test_empty_pipeline() -> None:
    pipeline = Pipeline([])
    assert pipeline.output_to_func == {}
    assert pipeline.topological_generations.root_args == []
    assert pipeline.topological_generations.function_lists == []

    with pytest.raises(TypeError, match="must be a `PipeFunc` or callable"):
        pipeline.add(1)  # type: ignore[arg-type]


def test_unhashable_defaults() -> None:
    @pipefunc(output_name="c", defaults={"b": []})
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", defaults={"b": {}})
    def g(a, b):
        return a + b

    pipeline = Pipeline([f])
    assert pipeline.defaults == {"b": []}

    # The problem should occur when using the default twice
    with pytest.raises(ValueError, match="Inconsistent default"):
        Pipeline([f, g])


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


def test_set_debug_and_profile() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    pipeline = Pipeline([f])
    assert not f.debug
    assert not f.profile
    pipeline.debug = True
    pipeline.profile = True
    assert pipeline["c"].debug
    assert pipeline["c"].profile


def test_simple_cache() -> None:
    @pipefunc(output_name="c", cache=True)
    def f(a, b):
        return a, b

    with pytest.raises(ValueError, match="Invalid cache type"):
        Pipeline([f], cache_type="not_exist")  # type: ignore[arg-type]
    pipeline = Pipeline([f], cache_type="simple")
    assert pipeline("c", a=1, b=2) == (1, 2)
    assert pipeline.cache is not None
    assert pipeline.cache.cache == {("c", (("a", 1), ("b", 2))): (1, 2)}
    pipeline.cache.clear()
    assert pipeline("c", a={"a": 1}, b=[2]) == ({"a": 1}, [2])
    assert pipeline.cache.cache == {("c", (("a", (("a", 1),)), ("b", (2,)))): ({"a": 1}, [2])}
    pipeline.cache.clear()
    assert pipeline("c", a={"a"}, b=[2]) == ({"a"}, [2])
    assert pipeline.cache.cache == {("c", (("a", ("a",)), ("b", (2,)))): ({"a"}, [2])}


def test_hybrid_cache_lazy_warning() -> None:
    @pipefunc(output_name="c", cache=True)
    def f(a, b):
        return a, b

    with pytest.warns(UserWarning, match="Hybrid cache uses function evaluation"):
        Pipeline([f], cache_type="hybrid", lazy=True)


def test_cache_non_root_args() -> None:
    @pipefunc(output_name="c", cache=True)
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", cache=True)
    def g(c, b):
        return c + b

    pipeline = Pipeline([f, g], cache_type="simple")
    # Won't populate cache because `c` is not a root argument
    assert pipeline("d", c=1, b=2) == 3
    assert pipeline.cache is not None
    assert pipeline.cache.cache == {}


def test_axis_in_root_args() -> None:
    # Test reaches the `output_name in visited` condition
    @pipefunc(output_name="c", mapspec="a[i] -> c[i]")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", mapspec="c[i] -> d[i]")
    def g(a, c):
        return a + c

    @pipefunc(output_name="e", mapspec="c[i], d[i] -> e[i]")
    def h(c, d):
        return c + d

    pipeline = Pipeline([f, g, h])
    assert pipeline.independent_axes_in_mapspecs("e") == {"i"}


def test_nesting_funcs_with_bound() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", bound={"b": "b2"})
    def g(c, b):
        return c + b

    pipeline = Pipeline([f, g])
    assert pipeline("d", a="a", b="b") == "abb2"
    pipeline.nest_funcs(["c", "d"], "d")  # type: ignore[arg-type]
    assert pipeline("d", a="a", b="b") == "abb2"
    assert len(pipeline.functions) == 1


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
    results = pipeline.map({"foo": {"a": [0, 1, 2]}, "b": 1})
    assert results["foo.c"].output.tolist() == [1, 2, 3]


def test_accessing_copied_pipefunc() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    pipeline = Pipeline([f])
    with pytest.raises(
        ValueError,
        match=re.escape("you can access that function via `pipeline['c']`"),
    ):
        pipeline.drop(f=f)


def test_pipeline_getitem_exception() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    pipeline = Pipeline([f])
    with pytest.raises(
        KeyError,
        match=re.escape("No function with output name `'d'` in the pipeline, only `['c']`"),
    ):
        pipeline["d"]


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
    results = pipeline.map(inputs=kwargs)
    assert results["baz.z"].output == 15
    assert pipeline(foo={"a": 1, "b": 2}, bar={"a": 3}, b=4) == 15


def test_default_resources_from_pipeline() -> None:
    @pipefunc(output_name="c", resources={"memory": "1GB", "num_cpus": 2})
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c

    pipeline1 = Pipeline([f, g], default_resources={"memory": "2GB", "num_cpus": 1})

    @pipefunc(output_name="e")
    def h(d):
        return d

    pipeline2 = Pipeline([h], default_resources={"memory": "3GB", "num_cpus": 3})

    @pipefunc(output_name="f", resources={"memory": "4GB", "num_cpus": 4})
    def i(e):
        return e

    pipeline3 = Pipeline([i])

    pipeline = pipeline1 | pipeline2 | pipeline3
    assert pipeline._default_resources is None
    assert isinstance(pipeline["c"].resources, Resources)
    assert isinstance(pipeline["d"].resources, Resources)
    assert isinstance(pipeline["e"].resources, Resources)
    assert isinstance(pipeline["f"].resources, Resources)

    assert pipeline["c"].resources.num_cpus == 2
    assert pipeline["c"].resources.memory == "1GB"
    assert pipeline["d"].resources.num_cpus == 1
    assert pipeline["d"].resources.memory == "2GB"
    assert pipeline["e"].resources.num_cpus == 3
    assert pipeline["e"].resources.memory == "3GB"
    assert pipeline["f"].resources.num_cpus == 4


def test_resources_variable():
    @pipefunc(output_name="c", resources_variable="resources", resources={"num_gpus": 8})
    def f_c(a, b, resources):  # noqa: ARG001
        return resources.num_gpus

    assert f_c(a=1, b=2) == 8

    pipeline = Pipeline([f_c])
    assert pipeline(a=1, b=2) == 8

    with pytest.raises(ValueError, match="Unexpected keyword arguments: `{'resources'}`"):
        f_c(a=1, b=2, resources={"num_gpus": 4})

    with pytest.raises(ValueError, match="Unused keyword arguments: `resources`"):
        pipeline(a=1, b=2, resources={"num_gpus": 4})


def test_resources_variable_nested_func():
    @pipefunc(output_name="c", resources_variable="resources", resources={"num_gpus": 8})
    def f_c(a, b, resources):  # noqa: ARG001
        return resources.num_gpus

    @pipefunc(output_name="d")
    def f_d(c):
        return c

    nf = NestedPipeFunc([f_c, f_d], output_name="d")
    assert nf.resources.num_gpus == 8
    assert nf(a=1, b=2) == 8

    pipeline = Pipeline([nf])
    assert pipeline(a=1, b=2) == 8


def test_sharing_defaults() -> None:
    @pipefunc(output_name="c", defaults={"b": 1}, cache=True)
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", cache=True)
    def g(b, c):
        return b + c

    pipeline = Pipeline([f, g], cache_type="simple")
    assert pipeline("d", a=1) == 3
    assert pipeline.cache is not None
    assert pipeline.cache.cache == {("c", (("a", 1), ("b", 1))): 2, ("d", (("a", 1), ("b", 1))): 3}
    assert pipeline.map(inputs={"a": 1})["d"].output == 3
    assert pipeline.map(inputs={"a": 1, "b": 2})["d"].output == 5


def test_resources_variable_with_callable_resources() -> None:
    @pipefunc(
        output_name="c",
        resources=lambda kwargs: Resources(num_gpus=kwargs["a"] + kwargs["b"]),
        resources_variable="resources",
    )
    def f_c(a, b, resources):  # noqa: ARG001
        return resources

    @pipefunc(output_name="d", resources=lambda kwargs: kwargs["c"])  # 'c' is the resources of f_c
    def f_d(c):
        assert isinstance(c, Resources)
        return c

    @pipefunc(
        output_name="e",
        resources=lambda kwargs: kwargs["d"],  # 'd' is the resources of f_c
        resources_variable="resources",
    )
    def f_e(d, resources):
        assert isinstance(resources, Resources)
        assert isinstance(d, Resources)
        assert resources == d
        return resources

    pipeline = Pipeline([f_c, f_d, f_e])
    r = pipeline(a=1, b=2)
    assert isinstance(r, Resources)
    assert r.num_gpus == 3

    with pytest.raises(
        ValueError,
        match="A `NestedPipeFunc` cannot have nested functions with callable `resources`.",
    ):
        NestedPipeFunc([f_c, f_d, f_e], output_name="e")


def test_resources_variable_in_nested_func_with_defaults() -> None:
    def resources_func(kwargs) -> Resources:  # noqa: ARG001
        msg = "Should not be called"
        raise ValueError(msg)

    @pipefunc("c", resources=resources_func)
    def f(a, b):
        return a + b

    @pipefunc("d", resources_variable="resources", resources=resources_func)
    def g(c, resources):  # noqa: ARG001
        assert isinstance(resources, Resources)
        return resources

    nf = NestedPipeFunc([f, g], output_name="d", resources={"num_gpus": 3})
    pipeline = Pipeline([nf], default_resources={"memory": "4GB", "num_gpus": 1})

    assert isinstance(pipeline["d"].resources, Resources)
    assert pipeline["d"].resources == Resources(num_gpus=3, memory="4GB")
    r = pipeline(a=1, b=2)
    assert isinstance(r, Resources)
    assert r.num_gpus == 3
    assert r.memory == "4GB"


def test_resources_func_with_variable() -> None:
    def resources_with_cpu(kwargs) -> Resources:
        num_cpus = kwargs["a"] + kwargs["b"]
        return Resources(num_cpus=num_cpus)

    @pipefunc(
        output_name="i",
        resources=resources_with_cpu,
        resources_variable="resources",
    )
    def j(a, b, resources):
        assert isinstance(resources, Resources)
        assert resources.num_cpus == a + b
        return a * b

    result = j(a=2, b=3)
    assert result == 6

    pipeline = Pipeline([j])
    result = pipeline(a=2, b=3)
    assert result == 6
    result = pipeline.map(inputs={"a": 2, "b": 3}, parallel=False)
    assert result["i"].output == 6


def test_with_resource_func_with_defaults():
    def resources_with_cpu(kwargs) -> Resources:
        num_cpus = kwargs["a"] + kwargs["b"]
        return Resources(num_cpus=num_cpus)

    @pipefunc(
        output_name="i",
        resources=resources_with_cpu,
        resources_variable="resources",
    )
    def j(a, b, resources):
        assert isinstance(resources, Resources)
        assert resources.num_cpus == a + b
        return resources

    pipeline = Pipeline([j], default_resources={"num_gpus": 5, "num_cpus": 1000})
    result = pipeline(a=2, b=3)
    assert result.num_cpus == 5
    assert result.num_gpus == 5
    result = pipeline.map(inputs={"a": 2, "b": 3}, parallel=False, storage="dict")
    assert result["i"].output.num_cpus == 5
    assert result["i"].output.num_gpus == 5


def test_unhashable_bound() -> None:
    @pipefunc(output_name="c", bound={"b": []})
    def f(a, b):
        return a, b

    assert f(a=1) == (1, [])
    pipeline = Pipeline([f])
    assert pipeline(a=1) == (1, [])


def test_mapping_over_default() -> None:
    @pipefunc(output_name="out", mapspec="a[i], b[i] -> out[i]", defaults={"b": [1, 2, 3]})
    def f(a, b):
        return a + b

    pipeline = Pipeline([f])
    r_map = pipeline.map(inputs={"a": [1, 2, 3]})
    assert r_map["out"].output.tolist() == [2, 4, 6]


def test_calling_add_with_autogen_mapspec():
    def foo(vector):
        return vector

    def bar(inpt, factor):
        return inpt * factor

    pipeline = Pipeline([])
    pipeline.add(PipeFunc(func=foo, output_name="foo_out"))
    pipeline.add(
        PipeFunc(
            func=bar,
            output_name="bar_out",
            renames={"inpt": "foo_out"},
            mapspec="foo_out[i], factor[i] -> bar_out[i]",
        ),
    )

    results = pipeline.map(
        inputs={"vector": [1, 2, 3], "factor": [1, 2, 3]},
        internal_shapes={"foo_out": (3,)},
    )
    assert results["bar_out"].output.tolist() == [1, 4, 9]


def test_parameterless_pipefunc() -> None:
    @pipefunc(output_name="c")
    def f():
        return 1

    assert f() == 1

    pipeline = Pipeline([f])
    assert pipeline() == 1
    assert pipeline.topological_generations.root_args == []
    assert pipeline.topological_generations.function_lists == [[pipeline["c"]]]
    r = pipeline.map({})
    assert r["c"].output == 1

    @pipefunc(output_name="d")
    def g():
        return 2

    @pipefunc(output_name="e")
    def h(c, d):
        return c + d

    pipeline = Pipeline([f, g, h])

    assert pipeline() == 3
    assert pipeline.topological_generations.root_args == []
    assert pipeline.topological_generations.function_lists == [
        [pipeline["c"], pipeline["d"]],
        [pipeline["e"]],
    ]
    r = pipeline.map({})
    assert r["e"].output == 3
