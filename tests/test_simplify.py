"""Tests for pipefunc.py."""

from __future__ import annotations

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc._pipeline._simplify import _combine_nodes, _identify_combinable_nodes


def test_identify_combinable_nodes():
    @pipefunc(output_name=("d", "e"))
    def f_d(b, g, x=1):
        pass

    @pipefunc(output_name=("g", "h"))
    def f_g(a, x=1):
        pass

    @pipefunc(output_name="gg")
    def f_gg(g):
        pass

    @pipefunc(output_name="i")
    def f_i(gg, b, e):
        pass

    pipeline = Pipeline([f_d, f_g, f_i, f_gg])

    # `conservatively_combine=False`
    combinable_nodes = _identify_combinable_nodes(
        pipeline["i"],
        pipeline.graph,
        pipeline.all_root_args,
        conservatively_combine=False,
    )
    assert combinable_nodes == {pipeline["gg"]: {pipeline["g"]}, pipeline["i"]: {pipeline["d"]}}
    simple = pipeline.simplified_pipeline(conservatively_combine=False)
    assert len(simple.functions) == 2
    assert repr(simple.functions[0]) == "NestedPipeFunc(pipefuncs=[PipeFunc(f_gg), PipeFunc(f_g)])"
    assert repr(simple.functions[1]) == "NestedPipeFunc(pipefuncs=[PipeFunc(f_i), PipeFunc(f_d)])"
    assert simple.functions[0].output_name == ("g", "gg")
    assert simple.functions[1].output_name == "i"

    # `conservatively_combine=True`
    combinable_nodes = _identify_combinable_nodes(
        pipeline["i"],
        pipeline.graph,
        pipeline.all_root_args,
        conservatively_combine=True,
    )
    assert combinable_nodes == {pipeline["gg"]: {pipeline["g"]}}
    simple = pipeline.simplified_pipeline(conservatively_combine=True)
    assert len(simple.functions) == 3
    assert repr(simple.functions[0]) == "PipeFunc(f_d)"
    assert repr(simple.functions[1]) == "PipeFunc(f_i)"
    assert repr(simple.functions[2]) == "NestedPipeFunc(pipefuncs=[PipeFunc(f_gg), PipeFunc(f_g)])"
    assert simple.functions[1].output_name == "i"
    assert simple.functions[2].output_name == ("g", "gg")


def test_conservatively_combine():
    @pipefunc(output_name="x")
    def f1(a):
        return a

    @pipefunc(output_name="y")
    def f2(b, x):
        return x * b

    @pipefunc(output_name="z")
    def f3(b, x, y):
        return x * y * b

    pipeline = Pipeline([f1, f2, f3])

    root_args = pipeline.all_root_args
    assert root_args == {"x": ("a",), "y": ("a", "b"), "z": ("a", "b")}

    # Test with conservatively_combine=True
    combinable_nodes_true = _identify_combinable_nodes(
        pipeline["z"],
        pipeline.graph,
        pipeline.all_root_args,
        conservatively_combine=True,
    )
    assert combinable_nodes_true == {}

    # Test simplified_pipeline with conservatively_combine=True
    with pytest.raises(ValueError, match="No combinable nodes found"):
        _simplified_pipeline_true = pipeline.simplified_pipeline(
            "z",
            conservatively_combine=True,
        )

    # Test with conservatively_combine=False
    combinable_nodes_false = _identify_combinable_nodes(
        pipeline["z"],
        pipeline.graph,
        pipeline.all_root_args,
        conservatively_combine=False,
    )
    assert combinable_nodes_false == {pipeline["z"]: {pipeline["y"]}}

    # Test simplified_pipeline with conservatively_combine=False
    simplified_pipeline_false = pipeline.simplified_pipeline(
        "z",
        conservatively_combine=False,
    )
    simplified_functions_false = simplified_pipeline_false.functions
    assert len(simplified_functions_false) == 2
    function_names_false = [f.__name__ for f in simplified_functions_false]
    assert "f1" in function_names_false
    assert "NestedPipeFunc_z" in function_names_false

    # Check that the combined function has the expected input and output arguments
    combined_f3 = next(f for f in simplified_functions_false if f.__name__ == "NestedPipeFunc_z")
    assert combined_f3.parameters == ("b", "x")
    assert combined_f3.output_name == "z"

    # Check that the simplified pipeline produces the same output as the original pipeline
    input_data = {"a": 2, "b": 3}
    original_output = pipeline("z", **input_data)
    simplified_output_false = simplified_pipeline_false("z", **input_data)
    assert original_output == simplified_output_false

    assert pipeline._func_node_colors() == ["C1", "C0", "C0"]


def test_identify_combinable_nodes2():
    def f1(a, b, c, d):
        return a + b + c + d

    def f2(a, b, e):
        return a + b + e

    def f3(a, b, f1):
        return a + b + f1

    def f4(f1, f3):
        return f1 + f3

    def f5(f1, f4):
        return f1 + f4

    def f6(b, f5):
        return b + f5

    def f7(a, f2, f6):
        return a + f2 + f6

    pipeline = Pipeline([f1, f2, f3, f4, f5, f6, f7])
    m = pipeline

    expected = {m["f6"]: {m["f1"], m["f3"], m["f4"], m["f5"]}}
    combinable_nodes = _identify_combinable_nodes(
        pipeline["f7"],
        pipeline.graph,
        pipeline.all_root_args,
    )
    simplified_combinable_nodes = _combine_nodes(combinable_nodes)
    assert simplified_combinable_nodes == expected

    # Test simplified_pipeline
    simplified_pipeline = pipeline.simplified_pipeline()
    assert (
        pipeline.unique_leaf_node.output_name
        == simplified_pipeline.unique_leaf_node.output_name
        == "f7"
    )
    simplified_functions = simplified_pipeline.functions

    # Check that the simplified pipeline has the expected number of functions
    assert len(simplified_functions) == 3

    # Check that the simplified pipeline has the expected function names
    function_names = [f.__name__ for f in simplified_functions]
    assert "f2" in function_names
    assert "NestedPipeFunc_f6" in function_names
    assert "f7" in function_names

    # Check that the combined function has the expected input and output arguments
    combined_f6 = next(f for f in simplified_functions if f.__name__ == "NestedPipeFunc_f6")
    assert combined_f6.parameters == ("a", "b", "c", "d")
    assert combined_f6.output_name == "f6"

    # Check that the simplified pipeline produces the same output as the original pipeline
    input_data = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    original_output = pipeline("f7", **input_data)
    simplified_output = simplified_pipeline("f7", **input_data)
    assert original_output == simplified_output


def test_exception_simplify_mapspec():
    @pipefunc(output_name="x", mapspec="a[i] -> x[i]")
    def f1(a):
        return a

    @pipefunc(output_name="y")
    def f2(x):
        return x

    pipeline = Pipeline([f1, f2])
    with pytest.raises(
        NotImplementedError,
        match="PipeFunc`s with `mapspec` cannot be simplified currently",
    ):
        pipeline.simplified_pipeline()
