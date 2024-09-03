"""Tests for pipefunc.Pipeline."""

from __future__ import annotations

import pickle
import re

import pytest

from pipefunc import NestedPipeFunc, PipeFunc, Pipeline, pipefunc
from pipefunc.exceptions import UnusedParametersError


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


def test_tuple_outputs() -> None:
    @pipefunc(
        output_name=("c", "_throw"),
        profile=True,
        debug=True,
        output_picker=dict.__getitem__,
    )
    def f_c(a, b):
        return {"c": a + b, "_throw": 1}

    @pipefunc(output_name=("d", "e"))
    def f_d(b, c, x=1):
        return b * c, 1

    @pipefunc(output_name=("g", "h"), output_picker=getattr)
    def f_g(c, e, x=1):
        from types import SimpleNamespace

        print(f"Called f_g with c={c} and e={e}")
        return SimpleNamespace(g=c + e, h=c - e)

    @pipefunc(output_name="i")
    def f_i(h, g):
        return h + g

    pipeline = Pipeline(
        [f_c, f_d, f_g, f_i],
        debug=True,
        profile=True,
    )
    f = pipeline.func("i")
    r = f.call_full_output(a=1, b=2, x=3)["i"]
    assert r == f(a=1, b=2, x=3)
    assert (
        pipeline.root_args("g")
        == pipeline.root_args("h")
        == pipeline.root_args(("g", "h"))
        == ("a", "b", "x")
    )
    assert pipeline.func(("g", "h"))(a=1, b=2, x=3).g == 4
    assert pipeline.func_dependencies("i") == [("c", "_throw"), ("d", "e"), ("g", "h")]
    assert pipeline.func_dependents("c") == [("d", "e"), ("g", "h"), "i"]

    assert (
        pipeline.func_dependencies("g")
        == pipeline.func_dependencies("h")
        == pipeline.func_dependencies(("g", "h"))
        == [("c", "_throw"), ("d", "e")]
    )

    f = pipeline.func(("g", "h"))
    r = f(a=1, b=2, x=3)
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


def test_full_output() -> None:
    @pipefunc(output_name="f1")
    def f1(a, b):
        return a + b

    @pipefunc(output_name=("f2i", "f2j"))
    def f2(f1):
        return 2 * f1, 1

    @pipefunc(output_name="f3")
    def f3(a, f2i):
        return a + f2i

    pipeline = Pipeline([f1, f2, f3])  # type: ignore[arg-type]
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
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    pipeline = Pipeline([f1])
    with pytest.raises(UnusedParametersError, match="Unused keyword arguments"):
        pipeline("c", a=1, b=2, doesnotexist=3)

    pipeline("c", a=1, b=2)


def test_handle_error() -> None:
    @pipefunc(output_name="c")
    def f1(a, b):
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


def test_output_picker_single_output() -> None:
    @pipefunc(output_name=("y",), output_picker=dict.__getitem__)
    def f(a, b):
        return {"y": a + b, "_throw": 1}

    pipeline = Pipeline([f])
    assert pipeline("y", a=1, b=2) == 3


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


def test_unhashable_bound() -> None:
    @pipefunc(output_name="c", bound={"b": []})
    def f(a, b):
        return a, b

    assert f(a=1) == (1, [])
    pipeline = Pipeline([f])
    assert pipeline(a=1) == (1, [])


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


def test_invalid_type_hints():
    @pipefunc(("y1", "y2"))
    def f(a: int, b: str) -> tuple[int, str]:
        return (a, b)

    @pipefunc("z")
    def g(
        y1: int,
        y2: float,  # Incorrect type hint (should be str)
    ) -> str:
        return f"{y1=}, {y2=}"

    with pytest.raises(
        TypeError,
        match="Inconsistent type annotations for",
    ):
        Pipeline([f, g])
