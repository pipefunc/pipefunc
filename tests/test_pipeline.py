"""Tests for pipefunc.Pipeline."""

from __future__ import annotations

import copy
import importlib.util
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

from pipefunc import NestedPipeFunc, PipeFunc, Pipeline, pipefunc
from pipefunc.exceptions import UnusedParametersError
from pipefunc.typing import Array  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path

has_psutil = importlib.util.find_spec("psutil") is not None
has_rich = importlib.util.find_spec("rich") is not None


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

    pipeline = Pipeline([f1, f2, f3], debug=True)

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
    f_nested.update_renames(
        {"a": "a2", "b": "b2"},
        overwrite=True,
        update_from="original",
    )
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

    pipeline = Pipeline([f1, f2, f3], debug=True)

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


@pytest.mark.skipif(not has_rich, reason="rich not installed")
def test_pipeline_info(capsys: pytest.CaptureFixture) -> None:
    @pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f2(b, c, xx):
        return b * c * xx

    @pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f1, f2, f3])

    assert pipeline.info() == {
        "inputs": ("a", "b", "x", "xx"),
        "outputs": ("e",),
        "intermediate_outputs": ("c", "d"),
        "required_inputs": ("a", "b", "xx"),
        "optional_inputs": ("x",),
    }
    pipeline.info(print_table=True)
    captured = capsys.readouterr()
    assert "Pipeline Info" in captured.out


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


@pytest.mark.skipif(not has_psutil, reason="psutil not installed")
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

    pipeline = Pipeline([pipe_func1, pipe_func2, pipe_func3], debug=True)

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
    assert pipeline.root_args(None) == ("a", "b", "x")
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


def test_function_pickling() -> None:
    from .helpers import pipeline_test_function

    # Get the _PipelineAsFunc instance from the pipeline
    func = pipeline_test_function.func("test_function")

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

    with pytest.raises(
        ValueError,
        match="Either `f` or `output_name` should be provided",
    ):
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
    assert pipeline.error_snapshot is None
    try:
        pipeline("c", a=1, b=2)
    except ValueError as e:
        msg = "Error occurred while executing function `f1(a=1, b=2)`"
        assert msg in str(e) or msg in str(e.__notes__)  # type: ignore[attr-defined]  # noqa: PT017
        # NOTE: with pytest.raises match="..." does not work
        # with add_note for some reason on my Mac, however,
        # on CI it works fine (Linux)...
    assert pipeline.error_snapshot is not None


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

    @pipefunc(
        output_name="c",
        defaults={"a": "a_new", "b": "b_new"},
        renames={"a": "b", "b": "a"},
    )
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

    @pipefunc(output_name="d", defaults={"b": {}})
    def g(a, b):
        return a + b

    pipeline = Pipeline([f])
    assert pipeline.defaults == {"b": []}

    # The problem should occur when using the default twice
    with pytest.raises(ValueError, match="Inconsistent default"):
        Pipeline([f, g])


@pytest.mark.skipif(not has_psutil, reason="psutil not installed")
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
        match=re.escape(
            "No function with output name `'d'` in the pipeline, only `['c']`",
        ),
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
    r = pipeline.map({}, parallel=False, storage="dict")
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
    r = pipeline.map({}, parallel=False, storage="dict")
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


class Unpicklable:
    def __init__(self, a) -> None:
        self.a = a

    def __getstate__(self):
        msg = "Unpicklable object"
        raise RuntimeError(msg)


def test_unpicklable_run(tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def f(a):
        return Unpicklable(a)

    @pipefunc(output_name="z")
    def g(y):
        return 1

    pipeline = Pipeline([f, g])

    r = pipeline.map({"a": 1}, parallel=False, storage="dict")
    assert isinstance(r["y"].output, Unpicklable)
    assert r["z"].output == 1


def test_unpicklable_run_with_mapspec():
    @pipefunc(output_name="y", mapspec="a[i] -> y[i]")
    def f(a):
        return Unpicklable(a)

    @pipefunc(output_name="z", mapspec="a[i] -> z[i]")
    def g(a):
        return a

    pipeline = Pipeline([f, g])
    inputs = {"a": [1, 2, 3, 4]}
    r = pipeline.map(
        inputs,
        executor=ThreadPoolExecutor(max_workers=2),
        parallel=True,
        storage="dict",
    )
    assert isinstance(r["y"].output, np.ndarray)
    assert r["z"].output.tolist() == [1, 2, 3, 4]


def test_duplicate_output_names() -> None:
    @pipefunc(output_name="y")
    def f(a):
        return a

    with pytest.raises(
        ValueError,
        match="The function with output name `'y'` already exists in the pipeline.",
    ):
        Pipeline([f, f])

    p = Pipeline([f])
    with pytest.raises(
        ValueError,
        match="The function with output name `'y'` already exists in the pipeline.",
    ):
        p.add(f)


def test_adding_duplicates_output_name_tuple() -> None:
    @pipefunc(output_name="y")
    def f(a):
        return a

    @pipefunc(output_name=("y", "y2"))
    def g(b):
        return b

    pipeline = Pipeline([f])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The function with output name `'y'` already exists in the pipeline (`f(...) → y`)",
        ),
    ):
        pipeline.add(g)


def test_double_output_then_iterate_over_single_axis():
    def f1(x, y):
        return x, y

    def f2(a):
        return 1

    pipeline = Pipeline(
        [
            PipeFunc(
                f1,
                ("a", "b"),
                mapspec="x[i], y[j] -> a[i, j], b[i, j]",
            ),
            PipeFunc(f2, "c", mapspec="a[:, j] -> c[j]"),
        ],
    )
    pipeline.map({"x": np.arange(3), "y": np.arange(3)}, parallel=False, storage="dict")
    assert pipeline.mapspec_axes == {
        "a": ("i", "j"),
        "b": ("i", "j"),
        "c": ("j",),
        "x": ("i",),
        "y": ("j",),
    }


@pytest.mark.parametrize("dim", [10, "?"])
def test_double_output_then_iterate_over_single_axis_gen_job(dim: int | Literal["?"]):
    def f1(x, y):
        return list(range(10)), list(range(10))

    def f2(a):
        return a

    pipeline = Pipeline(
        [
            PipeFunc(
                f1,
                ("a", "b"),
                mapspec="x[i], y[j] -> a[i, j, k], b[i, j, k]",
                internal_shape=(dim,),
            ),
            PipeFunc(f2, "c", mapspec="a[:, j, k] -> c[j, k]"),
        ],
    )
    results = pipeline.map(
        {"x": np.arange(3), "y": np.arange(3)},
        parallel=False,
        storage="dict",
    )
    assert results["c"].output.shape == (3, 10)


@dataclass
class Status:
    complete: list[int]
    incomplete: list[int]


def test_pipeline_map_zero_size() -> None:
    @pipefunc("status")
    def f1(mock_complete: list[int], mock_incomplete: list[int]) -> Status:
        return Status(mock_complete, mock_incomplete)

    @pipefunc("incomplete")
    def get_incomplete(status: Status) -> list[int]:
        return status.incomplete

    @pipefunc("completed")
    def load_complete(status: Status) -> list[int]:
        # Not actually doing anything
        return status.complete

    @pipefunc("executed", mapspec="incomplete[i] -> executed[i]")
    def run_incomplete(incomplete: int) -> int:
        return incomplete

    @pipefunc("result")
    def combine(completed: list[int], executed: Array[int]) -> list[int]:
        return completed + list(executed)

    pipeline = Pipeline([f1, get_incomplete, load_complete, run_incomplete, combine])
    result = pipeline.map(
        {"mock_complete": [0], "mock_incomplete": [1, 2, 3]},
        internal_shapes={"incomplete": ("?",)},
        parallel=False,
        storage="dict",
    )
    assert result["result"].output == [0, 1, 2, 3]
    # Now with empty complete
    result = pipeline.map(
        {"mock_complete": [], "mock_incomplete": [0, 1, 2, 3]},
        internal_shapes={"incomplete": ("?",)},
        parallel=False,
        storage="dict",
    )
    assert result["result"].output == [0, 1, 2, 3]

    # Now with empty incomplete
    # NOTE: Hits the `not args.missing and not args.existing` edge case
    result = pipeline.map(
        {"mock_complete": [0, 1, 2, 3], "mock_incomplete": []},
        internal_shapes={"incomplete": ("?",)},
        parallel=False,
        storage="dict",
    )
    assert result["result"].output == [0, 1, 2, 3]


def test_run_for_output_name_that_does_not_exist() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    pipeline = Pipeline([f])
    with pytest.raises(
        ValueError,
        match=re.escape("No function with output name `d` in the pipeline, only `c`."),
    ):
        pipeline("d", a=1, b=2)


def test_nested_pipefunc_in_pipeline_renames() -> None:
    @pipefunc(output_name="x")
    def fa(n: int) -> int:
        return 2 + n

    @pipefunc(output_name="y")
    def fb(x: int) -> int:
        return 2 * x

    # Run without nesting first
    pipeline_base = Pipeline([fa, fb], scope="test")
    assert pipeline_base.run("test.y", kwargs={"test.n": 2}) == 8

    pipeline = Pipeline([NestedPipeFunc([fa, fb], ("x", "y"))], scope="test")
    func = pipeline.output_to_func["test.x"]
    assert func.renames == {"n": "test.n", "x": "test.x", "y": "test.y"}
    assert isinstance(func, NestedPipeFunc)
    assert func.pipeline.functions[0].renames == {}
    r = pipeline.run("test.y", kwargs={"test.n": 2})
    assert r == 8
    r = pipeline.map(inputs={"test.n": 2}, parallel=False, storage="dict")
    assert r["test.y"].output == 8


def test_run_multiple_outputs() -> None:
    @pipefunc(output_name=("x", "y"))
    def f(a: int) -> tuple[int, int]:
        return a, 2 * a

    @pipefunc(output_name="z")
    def g(x: int, y: int) -> int:
        return x + y

    pipeline = Pipeline([f])
    r = pipeline.run(("x", "y"), kwargs={"a": 2}, full_output=True)
    # NOTE: Now unpacks the tuple but still also has the tuple.
    # Changed in #536 to fix a real issue.
    assert r == {"a": 2, ("x", "y"): (2, 4), "x": 2, "y": 4}

    pipeline = Pipeline([f, g])
    r = pipeline.run(("z"), kwargs={"a": 2}, full_output=True)
    assert r == {"a": 2, "x": 2, "y": 4, "z": 6}


def test_run_multiple_outputs_not_return_all() -> None:
    @pipefunc(output_name=("x", "y", "z"))
    def f(a: int) -> tuple[int, int, int]:
        return a, 2 * a, 3 * a

    pipeline = Pipeline([f])
    r = pipeline.run(("x", "y", "z"), kwargs={"a": 2}, full_output=True)
    assert r == {"a": 2, ("x", "y", "z"): (2, 4, 6), "x": 2, "y": 4, "z": 6}

    r2 = pipeline.run("x", kwargs={"a": 2}, full_output=True)
    assert r2 == {"a": 2, "x": 2, "y": 4, "z": 6}

    with pytest.raises(
        ValueError,
        match=re.escape("No function with output name `('x', 'y')` in the pipeline"),
    ):
        # This currently is not possible because we only allow string output OR
        # full exact tuple output_name.
        pipeline.run(("x", "y"), kwargs={"a": 2}, full_output=True)


def test_join_pipeline_preserves_defaults() -> None:
    @pipefunc(output_name="c", defaults={"b": 2})
    def f(a, b=1):
        return a + b

    @pipefunc(output_name="x.d", renames={"c": "x.c"})
    def g(c):
        return c + 1

    pipeline1 = Pipeline([f], scope="x")
    pipeline2 = Pipeline([g])
    pipeline = pipeline1.join(pipeline2)
    assert pipeline.run("x.d", kwargs={"x.a": 1}) == 4
    assert pipeline.defaults == {"x.b": 2}


def test_deepcopy_with_cache() -> None:
    @pipefunc(output_name="y")
    def f(x):
        return x

    pipeline = Pipeline([f], cache_type="hybrid")
    copy.deepcopy(pipeline)


def test_run_multiple_outputs_list() -> None:
    @pipefunc(output_name=("y1", "y2"))
    def f(a, b):
        return a + b, 1

    @pipefunc(output_name="z1")
    def g(a, b, y1):
        return a * b + y1

    @pipefunc(output_name="z2")
    def h(a, b, y2):
        return a * b + y2

    pipeline = Pipeline([f, g, h])
    assert len(pipeline.leaf_nodes) == 2
    y1 = 2 + 1
    y2 = 1
    assert pipeline.run(["z1", "z2"], kwargs={"a": 1, "b": 2}) == (
        1 * 2 + y1,
        1 * 2 + y2,
    )


def test_disjoint_pipefuncs() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(a, b):
        return a * b

    pipeline = Pipeline([f, g])
    assert pipeline.run(["c", "d"], kwargs={"a": 3, "b": 4}) == (7, 12)
    assert pipeline.func("c")(a=3, b=4) == 7
    assert pipeline.func("d")(a=3, b=4) == 12
    func = pipeline.func(["c", "d"])
    assert func(a=3, b=4) == (7, 12)
    func2 = pipeline.func(["d", "c"])
    assert func2(a=3, b=4) == (12, 7)


def test_run_allow_unused() -> None:
    @pipefunc(output_name="x")
    def fa(n: int, m: int = 0) -> int:
        return 2 + n + m

    @pipefunc(output_name="y")
    def fb(x: int, b: int) -> int:
        return 2 * x * b

    pipeline = Pipeline([fa, fb])
    assert pipeline.run(output_name="y", kwargs={"n": 1, "m": 2, "b": 3}) == 30
    with pytest.raises(UnusedParametersError, match="Unused keyword arguments: `b`."):
        pipeline.run(
            output_name="x",
            kwargs={"n": 1, "m": 2, "b": 3},
            allow_unused=False,
        )
    assert (
        pipeline.run(
            output_name="x",
            kwargs={"n": 1, "m": 2, "b": 3},
            allow_unused=True,
        )
        == 5
    )
