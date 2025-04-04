import re

import networkx as nx
import pytest

from pipefunc import ErrorSnapshot, NestedPipeFunc, Pipeline, VariantPipeline, pipefunc
from pipefunc.map._mapspec import ArraySpec
from pipefunc.resources import Resources
from pipefunc.typing import NoAnnotation


def test_nested_pipefunc_defaults() -> None:
    @pipefunc(output_name="c", defaults={"b": 2})
    def f(a, b: float):
        return a + b

    @pipefunc(output_name="d")
    def g(c) -> int:
        return c

    nf = NestedPipeFunc([f, g])
    pipeline = Pipeline([nf])
    assert nf.__name__ == "NestedPipeFunc_c_d"
    assert nf.defaults == {"b": 2}
    assert nf.output_name == ("c", "d")
    assert nf(a=1) == (3, 3)
    assert pipeline(a=1) == (3, 3)
    r = pipeline.map(inputs={"a": 1}, parallel=False, storage="dict")
    assert r["c"].output == 3
    assert r["d"].output == 3
    nf.update_defaults({"a": 5, "b": 10})
    # Need to do the same on the pipeline (since the nf is copied)
    pipeline["c"].update_defaults({"a": 5, "b": 10})
    assert nf.defaults == {"a": 5, "b": 10}
    assert nf() == (15, 15)
    assert nf.parameter_annotations == {"b": float}
    assert nf.output_annotation == {"c": NoAnnotation, "d": int}
    assert pipeline() == (15, 15)


def test_nested_pipefunc_multiple_outputs_defaults() -> None:
    @pipefunc(output_name=("e", "f"))
    def h(x, y=10):
        return x, y

    @pipefunc(output_name=("out1", "out2"))
    def i(e, f) -> tuple[int, int]:
        return e, f

    nf2 = NestedPipeFunc([h, i], output_name=("out1", "out2"))
    assert nf2.defaults == {"y": 10}
    assert nf2(x=5) == (5, 10)
    assert nf2.output_annotation == {"out1": int, "out2": int}


def test_nested_pipefunc_bound() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c

    nf = NestedPipeFunc([f, g], output_name="d")
    nf.update_bound({"a": 1})
    assert nf.parameters == ("a", "b")
    assert nf.bound == {"a": 1}
    assert nf.pipeline["c"].bound == {}
    assert nf(a=10, b=2) == 3  # a is bound to 1, so input a=10 is ignored
    nf.update_bound({"b": 5})
    assert nf.pipeline["c"].bound == {}
    assert nf.bound == {"a": 1, "b": 5}
    assert nf(a=100, b=200) == 6  # a and b are bound to 1 and 5 respectively


def test_nested_pipefunc_bound_in_nest() -> None:
    @pipefunc(output_name="c", bound={"b": 2})
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c

    nf = NestedPipeFunc([f, g], output_name="d")
    assert nf(a=1) == 3


def test_nested_pipefunc_bound_in_pipeline() -> None:
    @pipefunc(output_name="x", renames={"n": "n_"})
    def fa(n: int) -> int:
        return 2 + n

    @pipefunc(output_name="y", bound={"b_": 1}, defaults={"x": 1}, renames={"b": "b_"})
    def fb(x: int, b: int) -> int:
        return 2 * x * b

    nf = NestedPipeFunc([fa, fb], ("x", "y"))
    assert nf(n_=1) == (3, 6)
    assert nf.defaults == {}
    assert nf.bound == {}
    pipeline_nested_test = Pipeline([nf])
    assert pipeline_nested_test(n_=1)
    with pytest.raises(ValueError, match=re.escape("Unexpected keyword arguments")):
        # if the child functions have bound, they are not parameters of the nested pipefunc!
        nf(n_=1, b_=10000000)
    assert nf.pipeline["y"](x=3) == 6
    assert nf(n_=1) == (3, 6)
    assert pipeline_nested_test.topological_generations.root_args == ["n_"]
    with pytest.raises(
        ValueError,
        match=re.escape("Got extra inputs: `b_` that are not accepted by this pipeline"),
    ):
        pipeline_nested_test.map(inputs={"n_": 1, "b_": 10000000}, parallel=False, storage="dict")
    r = pipeline_nested_test.map(inputs={"n_": 1}, parallel=False, storage="dict")
    assert r["x"].output == 3
    assert r["y"].output == 6
    r = pipeline_nested_test.map(inputs={"n_": 1}, parallel=False, storage="dict")
    assert r["x"].output == 3
    assert r["y"].output == 6


def test_nested_pipefunc_multiple_outputs_bound() -> None:
    # Test with multiple outputs
    @pipefunc(output_name=("e", "f"))
    def h(x, y):
        return x, y

    @pipefunc(output_name=("out1", "out2"))
    def i(e, f):
        return e, f

    nf2 = NestedPipeFunc([h, i], output_name=("e", "out2"))
    nf2.update_bound({"x": 1})
    assert nf2.bound == {"x": 1}
    assert nf2.pipeline["e"].bound == {}
    assert nf2(x=5, y=10) == (1, 10)


def test_nested_pipefunc_mapspec() -> None:
    @pipefunc(output_name="c", mapspec="a[i], b[i] -> c[i]")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", mapspec="c[i] -> d[i]")
    def g(c):
        return c

    nf = NestedPipeFunc([f, g])
    assert nf.mapspec is not None
    assert nf.mapspec.input_indices == {"i"}
    assert nf.mapspec.output_indices == ("i",)
    assert nf.mapspec.inputs == (
        ArraySpec(name="a", axes=("i",)),
        ArraySpec(name="b", axes=("i",)),
    )
    assert nf.mapspec.outputs == (
        ArraySpec(name="c", axes=("i",)),
        ArraySpec(name="d", axes=("i",)),
    )

    # Test with different mapspecs
    @pipefunc(output_name="e", mapspec="x[i], y[j] -> e[i, j]")
    def h(x, y):
        return x * y

    @pipefunc(output_name="f", mapspec="e[i, j] -> f[i, j]")
    def k(e):
        return e - 1

    nf2 = NestedPipeFunc([h, k])
    assert nf2.mapspec is not None
    assert nf2.mapspec.input_indices == {"i", "j"}
    assert nf2.mapspec.output_indices == ("i", "j")
    assert str(nf2.mapspec) == "x[i], y[j] -> e[i, j], f[i, j]"

    # Test with custom mapspec
    nf3 = NestedPipeFunc([f, g], output_name="d", mapspec="a[i], b[i] -> d[i]")
    assert nf3.mapspec is not None
    assert nf3.mapspec.input_indices == {"i"}
    assert nf3.mapspec.output_indices == ("i",)


def test_nested_pipefunc_resources() -> None:
    @pipefunc(output_name="c", resources=Resources(cpus=2, memory="1GB"))
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", resources=Resources(cpus=1, memory="2GB"))
    def g(c):
        return c

    nf = NestedPipeFunc([f, g])
    assert nf.resources is not None
    assert isinstance(nf.resources, Resources)
    assert nf.resources.cpus == 2
    assert nf.resources.memory == "2GB"

    # Test with resources specified in NestedPipeFunc
    nf2 = NestedPipeFunc(
        [f, g],
        resources=Resources(cpus=4, memory="4GB"),
    )
    assert nf2.resources is not None
    assert isinstance(nf2.resources, Resources)
    assert nf2.resources.cpus == 4
    assert nf2.resources.memory == "4GB"


def test_nested_pipefunc_variants() -> None:
    @pipefunc(output_name="c", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant="sub")
    def f2(a, b):
        return a - b

    @pipefunc(output_name="d")
    def g(c):
        return c

    vp = VariantPipeline([f, f2, g])
    nf = NestedPipeFunc(
        [vp.with_variant("add").functions[0], g],
        variant="add",
    )
    assert nf.variant == {None: "add"}
    nf2 = NestedPipeFunc(
        [vp.with_variant("sub").functions[0], g],
        variant="sub",
    )
    assert nf2.variant == {None: "sub"}

    vp = VariantPipeline([nf, nf2])
    pipeline_add = vp.with_variant("add")
    assert isinstance(pipeline_add, Pipeline)
    assert pipeline_add(a=1, b=2) == (3, 3)
    pipeline_sub = vp.with_variant("sub")
    assert isinstance(pipeline_sub, Pipeline)
    assert pipeline_sub(a=1, b=2) == (-1, -1)


def test_nested_pipefunc_variant_groups() -> None:
    @pipefunc(output_name="c", variant={"op": "add"})
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant={"op": "sub"})
    def f2(a, b):
        return a - b

    @pipefunc(output_name="d", variant={"mult": "yes"})
    def g(c):
        return c * 2

    @pipefunc(output_name="d", variant={"mult": "no"})
    def g2(c):
        return c

    vp = VariantPipeline([f, f2, g, g2])
    nf = NestedPipeFunc(
        [
            vp.with_variant({"op": "add", "mult": "yes"}).functions[0],
            vp.with_variant({"op": "add", "mult": "yes"}).functions[1],
        ],
        variant={"op_mult": "add_yes"},
    )

    nf2 = NestedPipeFunc(
        [
            vp.with_variant({"op": "sub", "mult": "no"}).functions[0],
            vp.with_variant({"op": "sub", "mult": "no"}).functions[1],
        ],
        variant={"op_mult": "sub_no"},
    )

    vp = VariantPipeline([nf, nf2])
    pipeline1 = vp.with_variant({"op_mult": "add_yes"})
    pipeline2 = vp.with_variant({"op_mult": "sub_no"})
    assert isinstance(pipeline1, Pipeline)
    assert isinstance(pipeline2, Pipeline)
    assert pipeline1(a=1, b=2) == (3, 6)
    assert pipeline2(a=1, b=2) == (-1, -1)


def test_nested_pipefunc_with_scope() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c

    nf = NestedPipeFunc([f, g], output_name="d")
    nf.update_scope("my_scope", "*", "*")
    assert nf.parameters == ("my_scope.a", "my_scope.b")
    assert nf.output_name == "my_scope.d"
    assert nf(my_scope={"a": 1, "b": 2}) == 3


def test_nested_pipefunc_output_picker() -> None:
    @pipefunc(
        output_name=("c", "d"),
        output_picker=lambda output, key: output[0] if key == "c" else output[1],
    )
    def h(a, b):
        return a + b, a * b

    @pipefunc(output_name="e")
    def i(c, d):
        return c + d

    nf = NestedPipeFunc([h, i], output_name="e")
    assert nf(a=1, b=2) == 5


def test_nested_pipefunc_error_snapshot() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        msg = "Intentional error"
        raise ValueError(msg)

    nf = NestedPipeFunc([f, g])
    with pytest.raises(ValueError, match="Intentional error"):
        nf(a=1, b=2)
    assert nf.error_snapshot is not None
    assert isinstance(nf.error_snapshot, ErrorSnapshot)
    assert nf.error_snapshot.args == ()
    assert nf.error_snapshot.kwargs == {"a": 1, "b": 2}


def test_nested_pipefunc_no_unique_leaf_node() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(a, b):
        return a + b

    nf = NestedPipeFunc([f, g])
    assert nf.output_name == ("c", "d")
    assert nf(a=1, b=2) == (3, 3)


def test_nested_pipefunc_variant_different_output_name() -> None:
    @pipefunc(output_name="sum_", variant={"op": "add"})
    def f(a, b):
        return a + b

    @pipefunc(output_name="diff", variant={"op": "sub"})
    def f2(a, b):
        return a - b

    @pipefunc(output_name="double")
    def g(sum_):
        return sum_ * 2

    @pipefunc(output_name="half")
    def g2(diff):
        return diff / 2

    vp = VariantPipeline([f, f2, g, g2])
    vp_add = vp.with_variant({"op": "add"})
    vp_sub = vp.with_variant({"op": "sub"})

    nf = NestedPipeFunc(
        [vp_add.functions[0], vp_add.functions[1]],
        variant={"op": "add"},
        output_name="double",
    )
    nf2 = NestedPipeFunc(
        [vp_sub.functions[0], vp_sub.functions[2]],
        variant={"op": "sub"},
        output_name="half",
    )

    vp2 = VariantPipeline([nf, nf2])
    pipeline1 = vp2.with_variant({"op": "add"})
    assert isinstance(pipeline1, Pipeline)
    assert pipeline1(a=1, b=2) == 6
    pipeline2 = vp2.with_variant({"op": "sub"})
    assert isinstance(pipeline2, Pipeline)
    assert pipeline2(a=1, b=2) == -0.5


def test_nested_pipefunc_output_annotation() -> None:
    @pipefunc(output_name="c")
    def f1(a, b) -> int:
        return a + b

    @pipefunc(output_name="d")
    def f2(c) -> float:
        return c / 2

    funcs = [f1, f2]
    nf1 = NestedPipeFunc(funcs, output_name="d")
    assert nf1.output_annotation == {"d": float}
    nf1 = NestedPipeFunc(funcs, output_name="c")
    assert nf1.output_annotation == {"c": int}
    nf1 = NestedPipeFunc(funcs, output_name=("c", "d"))
    assert nf1.output_annotation == {"c": int, "d": float}


def test_join_pipeline_with_nested_preserves_defaults() -> None:
    @pipefunc(output_name="c", defaults={"b": 2})
    def f(a, b=1):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c + 1

    @pipefunc(output_name="e")
    def h(d):
        return d + 1

    pipeline1 = Pipeline([NestedPipeFunc([f, g], output_name="d")])
    pipeline1.update_renames({"b": "scope.b"})
    assert pipeline1["d"].renames == {"b": "scope.b"}
    assert pipeline1["d"].defaults == {"scope.b": 2}
    assert pipeline1.defaults == {"scope.b": 2}
    assert pipeline1("d", a=1) == 4
    r = pipeline1.map(inputs={"a": 1}, parallel=False, storage="dict")
    assert r["d"].output == 4
    assert pipeline1.info() == {  # Should not have "c"
        "required_inputs": ("a",),
        "optional_inputs": ("scope.b",),
        "inputs": ("a", "scope.b"),
        "intermediate_outputs": (),
        "outputs": ("d",),
    }
    pipeline2 = Pipeline([h])
    pipeline = pipeline1.join(pipeline2)
    assert pipeline["d"].renames == {"b": "scope.b"}
    assert pipeline["d"].defaults == {"scope.b": 2}
    assert pipeline.defaults == {"scope.b": 2}
    assert pipeline("e", a=1) == 5
    r = pipeline.map(inputs={"a": 1}, parallel=False, storage="dict")
    assert r["e"].output == 5


def test_bound_inside_nested_pipefunc_and_other_function_uses_same_parameter() -> None:
    @pipefunc(output_name="c", bound={"b": 2})
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(b, c):
        return b + c

    pipeline = Pipeline([f, g])
    assert pipeline(a=1, b=1) == (1 + (1 + 2)) == 4
    r = pipeline.map(inputs={"a": 1, "b": 1}, parallel=False, storage="dict")
    assert r["d"].output == 4
    nf = NestedPipeFunc([f, g], output_name="d")
    assert nf(a=1, b=1) == 1 + (1 + 2) == 4
    pipeline2 = Pipeline([nf])
    assert pipeline2(a=1, b=1) == 4
    r = pipeline2.map(inputs={"a": 1, "b": 1}, parallel=False, storage="dict")
    assert r["d"].output == 4
    with pytest.raises(ValueError, match=re.escape("Missing value for argument `b`")):
        pipeline2(a=1)
    with pytest.raises(
        ValueError,
        match=re.escape("Missing inputs: `b`"),
    ):
        pipeline2.map(inputs={"a": 1}, parallel=False, storage="dict")
    pipeline2.update_defaults({"b": 10})
    assert pipeline2(a=1) == 10 + (1 + 2) == 13
    r = pipeline2.map(inputs={"a": 1}, parallel=False, storage="dict")
    assert r["d"].output == 13


@pytest.mark.parametrize("scope", ["scope.", ""])
def test_nest_bound(scope: str) -> None:
    @pipefunc(output_name="x")
    def fa(n: int, m: int = 0) -> int:
        return 2 + n + m

    @pipefunc(output_name="y", bound={"b": 1})
    def fb(x: int, b: int) -> int:
        return 2 * x * b

    pipeline_nested_test = Pipeline(
        [
            NestedPipeFunc([fa, fb], ("x", "y"), function_name="my function"),
        ],
        scope=scope[:-1] if scope else None,
    )
    y = 2 * (2 + 4 + 0) * 1
    assert pipeline_nested_test.run(f"{scope}y", kwargs={f"{scope}n": 4}) == y
    r = pipeline_nested_test.map(inputs={f"{scope}n": 4}, parallel=False, storage="dict")
    assert r[f"{scope}y"].output == y
    with pytest.raises(ValueError, match=re.escape(f"Missing value for argument `{scope}n`")):
        pipeline_nested_test.run(f"{scope}y", kwargs={})
    with pytest.raises(
        ValueError,
        match=re.escape(f"Missing inputs: `{scope}n`."),
    ):
        pipeline_nested_test.map(inputs={}, parallel=False, storage="dict")


def test_annotations_nested_pipefunc() -> None:
    @pipefunc(output_name="c")
    def f(a: int, b: int) -> int:
        return a + b

    @pipefunc(output_name="d")
    def g(c: int) -> int:
        return c

    nf = NestedPipeFunc([f, g])
    assert nf.parameter_annotations == {"a": int, "b": int}
    assert nf.output_annotation == {"c": int, "d": int}
    nf2 = NestedPipeFunc([f, g], output_name="d")
    assert nf2.parameter_annotations == {"a": int, "b": int}
    assert nf2.output_annotation == {"d": int}


def test_nested_pipefunc_with_scoped_pipefuncs() -> None:
    @pipefunc(output_name="c", scope="foo")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", scope="foo")
    def g(c):
        return c

    nf = NestedPipeFunc([f, g])
    assert nf.parameter_scopes == {"foo"}
    assert nf(foo={"a": 1, "b": 2}) == (3, 3)


def test_disjoint_nested_pipefuncs() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(a, b):
        return a * b

    nested = NestedPipeFunc([f, g])
    assert nested.parameters == ("a", "b")
    assert nested.output_name == ("c", "d")
    pipeline = Pipeline([nested])
    r = pipeline.map(inputs={"a": 3, "b": 4}, parallel=False, storage="dict")
    assert r["c"].output == 7
    assert r["d"].output == 12


def test_disjoint_nested_pipefuncs_multiple_ouputs() -> None:
    @pipefunc(output_name=("c1", "c2"))
    def f(a, b):
        return 2 * a, 2 * b

    @pipefunc(output_name=("d1", "d2"))
    def g(a, b):
        return a * b, a + b

    nested = NestedPipeFunc([f, g])
    assert nested.parameters == ("a", "b")
    assert nested.output_name == ("c1", "c2", "d1", "d2")
    pipeline = Pipeline([nested])
    r = pipeline.map(inputs={"a": 3, "b": 4}, parallel=False, storage="dict")
    assert r["c1"].output == 6
    assert r["c2"].output == 8
    assert r["d1"].output == 12
    assert r["d2"].output == 7


def test_disjoint_nested_pipefuncs_multiple_ouputs_mixed() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name=("d1", "d2"))
    def g(a, b):
        return a * b, a + b

    nested = NestedPipeFunc([f, g])
    assert nested.parameters == ("a", "b")
    assert nested.output_name == ("c", "d1", "d2")
    pipeline = Pipeline([nested])
    r = pipeline.map(inputs={"a": 3, "b": 4}, parallel=False, storage="dict")
    assert r["c"].output == 7
    assert r["d1"].output == 12
    assert r["d2"].output == 7


def test_linear_pipeline_nest_outer_funcs_error() -> None:
    @pipefunc(output_name="x")
    def f():
        return 1

    @pipefunc(output_name="y")
    def g(x):
        return x + 1

    @pipefunc(output_name="z")
    def h(y):
        return y * 2

    pipeline = Pipeline([f, g, h])
    r = pipeline.map(inputs={}, parallel=False, storage="dict")
    assert r["z"].output == 4
    assert pipeline() == 4

    # Should not be possible to nest functions that have a dependency in the middle of the pipeline
    with pytest.raises(
        nx.exception.NetworkXUnfeasible,
        match="Graph contains a cycle or graph changed during iteration",
    ):
        pipeline.nest_funcs({"x", "z"})


def test_nested_pipefunc_with_multiple_outputs_then_adding_scope_annotation() -> None:
    # Tests regression fixed in https://github.com/pipefunc/pipefunc/pull/695
    @pipefunc(output_name=("c", "d"))
    def f(a: int, b: int) -> tuple[int, int]:
        return a + b, a * b

    @pipefunc(output_name="e")
    def g(c: int, d: int) -> int:
        return c + d

    # Validating that the pipeline has correct output annotations
    normal_pipeline = Pipeline([f, g], scope="foo")
    assert normal_pipeline.output_annotations == {"foo.e": int, "foo.c": int, "foo.d": int}
    assert normal_pipeline.all_output_names == {"foo.c", "foo.d", "foo.e"}
    assert normal_pipeline.root_args() == ("foo.a", "foo.b")

    nf = NestedPipeFunc([f, g])
    pipeline = Pipeline([nf])
    assert pipeline("e", a=1, b=2) == 5
    assert pipeline.output_annotations == {"e": int, "c": int, "d": int}
    pipeline.update_scope("foo", inputs="*", outputs="*")
    assert pipeline.root_args() == ("foo.a", "foo.b")
    assert pipeline.all_output_names == {"foo.c", "foo.d", "foo.e"}
    assert pipeline.output_annotations == {"foo.e": int, "foo.c": int, "foo.d": int}

    pipeline2 = Pipeline([nf], scope="foo")
    assert pipeline2.output_annotations == {"foo.e": int, "foo.c": int, "foo.d": int}


def test_nested_pipefunc_single_output_with_scope() -> None:
    """Test that a nested pipefunc with a single output correctly handles scopes."""

    @pipefunc(output_name="c")
    def f(a: int, b: float) -> float:
        return a + b

    @pipefunc(output_name="d")
    def g(c: float) -> int:
        return int(c)

    # Create nested pipefunc with single output
    nf = NestedPipeFunc([f, g], output_name="d")
    assert nf.output_annotation == {"d": int}

    # Add scope directly to nested pipefunc
    nf.update_scope("my_scope", inputs="*", outputs="*")
    assert nf.output_annotation == {"my_scope.d": int}

    # We don't test parameter_annotations here because it doesn't
    # handle scopes in the same way as output_annotation yet

    # Verify execution still works
    assert nf(my_scope={"a": 1, "b": 2.5}) == 3

    # Create pipeline with the nested pipefunc
    pipeline = Pipeline([nf])
    assert pipeline.output_annotations == {"my_scope.d": int}


def test_nested_pipefunc_different_scopes_for_outputs() -> None:
    """Test that a nested pipefunc correctly handles different scopes for different outputs."""

    @pipefunc(output_name=("c", "d"))
    def f(a: int, b: int) -> tuple[float, str]:
        return a + b + 0.5, f"sum:{a + b}"

    @pipefunc(output_name=("e", "f"))
    def g(c: float, d: str) -> tuple[int, str]:
        return int(c), f"processed:{d}"

    # Create nested pipefunc with multiple outputs
    nf = NestedPipeFunc([f, g])

    # Test output annotations before scopes
    assert nf.output_annotation == {"c": float, "d": str, "e": int, "f": str}

    # Create pipeline with the nested pipefunc
    pipeline = Pipeline([nf])

    # Apply different scopes to different outputs
    pipeline.update_scope("scope1", outputs={"c", "e"})
    pipeline.update_scope("scope2", outputs={"d", "f"})

    # Check that output annotations reflect the different scopes
    assert pipeline.output_annotations == {
        "scope1.c": float,
        "scope2.d": str,
        "scope1.e": int,
        "scope2.f": str,
    }

    # Verify execution
    r = pipeline.map(inputs={"a": 1, "b": 2}, parallel=False, storage="dict")
    assert r["scope1.c"].output == 3.5
    assert r["scope2.d"].output == "sum:3"
    assert r["scope1.e"].output == 3
    assert r["scope2.f"].output == "processed:sum:3"


def test_nested_pipefunc_scope_removal() -> None:
    """Test that a nested pipefunc correctly handles scope removal."""

    @pipefunc(output_name="c", scope="original")
    def f(a: int, b: int) -> int:
        return a + b

    @pipefunc(output_name="d", scope="original")
    def g(c: int) -> int:
        return c * 2

    # Create nested pipefunc with scoped functions
    nf = NestedPipeFunc([f, g])
    assert nf.output_annotation == {"original.c": int, "original.d": int}

    # Create pipeline with the nested pipefunc
    pipeline = Pipeline([nf])
    assert pipeline.output_annotations == {"original.c": int, "original.d": int}

    # Remove scopes
    pipeline.update_scope(None, inputs="*", outputs="*")

    # Check that output annotations reflect the removed scopes
    assert pipeline.output_annotations == {"c": int, "d": int}

    # Verify execution works with unscoped parameters
    assert pipeline(a=2, b=3) == (5, 10)
