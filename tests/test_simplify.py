"""Tests for pipefunc.py."""

from __future__ import annotations

from pipefunc import Pipeline, pipefunc
from pipefunc._simplify import _combine_nodes, _get_signature


def test_identify_combinable_nodes():
    @pipefunc(output_name=("d", "e"))
    def f_d(b, g, x=1):  # noqa: ARG001
        pass

    @pipefunc(output_name=("g", "h"))
    def f_e(a, x=1):  # noqa: ARG001
        pass

    @pipefunc(output_name="gg")
    def f_gg(g):  # noqa: ARG001
        pass

    @pipefunc(output_name="i")
    def f_i(gg, b, e):  # noqa: ARG001
        pass

    pipeline = Pipeline([f_d, f_e, f_i, f_gg])
    combinable_nodes = pipeline._identify_combinable_nodes(
        "i",
        conservatively_combine=True,
    )
    assert combinable_nodes == {f_gg: {f_e}}
    sig_in, sig_out = _get_signature(combinable_nodes, pipeline.graph)
    assert sig_in == {f_gg: {"a", "x"}}
    assert sig_out == {f_gg: {"gg", "h", "g"}}
    combinable_nodes = pipeline._identify_combinable_nodes(
        "i",
        conservatively_combine=False,
    )
    assert combinable_nodes == {f_gg: {f_e}, f_i: {f_d}}
    sig_in, sig_out = _get_signature(combinable_nodes, pipeline.graph)
    assert sig_in == {f_gg: {"a", "x"}, f_i: {"g", "b", "gg", "x"}}
    assert sig_out == {f_gg: {"gg", "h", "g"}, f_i: {"d", "i"}}


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

    pipeline = Pipeline([f1, f2, f3], debug=True, profile=True)

    root_args = pipeline.all_root_args
    assert root_args == {"x": ("a",), "y": ("a", "b"), "z": ("a", "b")}

    # Test with conservatively_combine=True
    combinable_nodes_true = pipeline._identify_combinable_nodes(
        "z",
        conservatively_combine=True,
    )
    assert combinable_nodes_true == {}

    # Test simplified_pipeline with conservatively_combine=True
    simplified_pipeline_true = pipeline.simplified_pipeline(
        "z",
        conservatively_combine=True,
    )
    simplified_functions_true = simplified_pipeline_true.functions
    assert len(simplified_functions_true) == 3
    function_names_true = [f.__name__ for f in simplified_functions_true]
    assert "f1" in function_names_true
    assert "f2" in function_names_true
    assert "f3" in function_names_true

    # Test with conservatively_combine=False
    combinable_nodes_false = pipeline._identify_combinable_nodes(
        "z",
        conservatively_combine=False,
    )
    assert combinable_nodes_false == {f3: {f2}}

    # Test simplified_pipeline with conservatively_combine=False
    simplified_pipeline_false = pipeline.simplified_pipeline(
        "z",
        conservatively_combine=False,
    )
    simplified_functions_false = simplified_pipeline_false.functions
    assert len(simplified_functions_false) == 2
    function_names_false = [f.__name__ for f in simplified_functions_false]
    assert "f1" in function_names_false
    assert "combined_f3" in function_names_false

    # Check that the combined function has the expected input and output arguments
    combined_f3 = next(f for f in simplified_functions_false if f.__name__ == "combined_f3")
    assert combined_f3.parameters == ["b", "x"]
    assert combined_f3.output_name == "z"

    # Check that the simplified pipeline produces the same output as the original pipeline
    input_data = {"a": 2, "b": 3}
    original_output = pipeline("z", **input_data)
    simplified_output_true = simplified_pipeline_true("z", **input_data)
    simplified_output_false = simplified_pipeline_false("z", **input_data)
    assert original_output == simplified_output_true == simplified_output_false

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
    m = pipeline.node_mapping

    expected = {m["f6"]: {m["f1"], m["f3"], m["f4"], m["f5"]}}
    combinable_nodes = pipeline._identify_combinable_nodes("f7")
    simplified_combinable_nodes = _combine_nodes(combinable_nodes)
    assert simplified_combinable_nodes == expected

    sig_in, sig_out = _get_signature(simplified_combinable_nodes, pipeline.graph)
    assert sig_in == {m["f6"]: {"a", "b", "c", "d"}}
    assert sig_out == {m["f6"]: {"f6"}}

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
    assert "combined_f6" in function_names
    assert "f7" in function_names

    # Check that the combined function has the expected input and output arguments
    combined_f6 = next(f for f in simplified_functions if f.__name__ == "combined_f6")
    assert combined_f6.parameters == ["a", "b", "c", "d"]
    assert combined_f6.output_name == "f6"

    # Check that the simplified pipeline produces the same output as the original pipeline
    input_data = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    original_output = pipeline("f7", **input_data)
    simplified_output = simplified_pipeline("f7", **input_data)
    assert original_output == simplified_output
