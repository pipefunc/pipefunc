import pytest

from pipefunc import ErrorSnapshot, NestedPipeFunc, Pipeline, VariantPipeline, pipefunc
from pipefunc.map._mapspec import ArraySpec
from pipefunc.resources import Resources


def test_nested_pipefunc_defaults() -> None:
    @pipefunc(output_name="c", defaults={"b": 2})
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c

    nf = NestedPipeFunc([f, g])
    assert nf.defaults == {"b": 2}
    assert nf.output_name == ("c", "d")
    assert nf(a=1) == (3, 3)
    nf.update_defaults({"a": 5, "b": 10})
    assert nf.defaults == {"a": 5, "b": 10}
    assert nf() == (15, 15)


def test_nested_pipefunc_multiple_outputs_defaults() -> None:
    @pipefunc(output_name=("e", "f"))
    def h(x, y=10):
        return x, y

    @pipefunc(output_name=("out1", "out2"))
    def i(e, f):
        return e, f

    nf2 = NestedPipeFunc([h, i], output_name=("out1", "out2"))
    assert nf2.defaults == {"y": 10}
    assert nf2(x=5) == (5, 10)


def test_nested_pipefunc_bound() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c

    nf = NestedPipeFunc([f, g], output_name="d")
    nf.update_bound({"a": 1})
    assert nf.bound == {"a": 1}
    assert nf(a=10, b=2) == 3  # a is bound to 1, so input a=10 is ignored
    nf.update_bound({"b": 5})
    assert nf.bound == {"a": 1, "b": 5}
    assert nf(a=100, b=200) == 6  # a and b are bound to 1 and 5 respectively


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
    assert nf.variant == "add"
    nf2 = NestedPipeFunc(
        [vp.with_variant("sub").functions[0], g],
        variant="sub",
    )
    assert nf2.variant == "sub"

    vp = VariantPipeline([nf, nf2])
    pipeline_add = vp.with_variant("add")
    assert isinstance(pipeline_add, Pipeline)
    assert pipeline_add(a=1, b=2) == (3, 3)
    pipeline_sub = vp.with_variant("sub")
    assert isinstance(pipeline_sub, Pipeline)
    assert pipeline_sub(a=1, b=2) == (-1, -1)


def test_nested_pipefunc_variant_groups() -> None:
    @pipefunc(output_name="c", variant_group="op", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op", variant="sub")
    def f2(a, b):
        return a - b

    @pipefunc(output_name="d", variant_group="mult", variant="yes")
    def g(c):
        return c * 2

    @pipefunc(output_name="d", variant_group="mult", variant="no")
    def g2(c):
        return c

    vp = VariantPipeline([f, f2, g, g2])
    nf = NestedPipeFunc(
        [
            vp.with_variant({"op": "add", "mult": "yes"}).functions[0],
            vp.with_variant({"op": "add", "mult": "yes"}).functions[1],
        ],
        variant_group="op_mult",
        variant="add_yes",
    )

    nf2 = NestedPipeFunc(
        [
            vp.with_variant({"op": "sub", "mult": "no"}).functions[0],
            vp.with_variant({"op": "sub", "mult": "no"}).functions[1],
        ],
        variant_group="op_mult",
        variant="sub_no",
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


def test_nested_pipefunc_no_leaf_node() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(a, b):
        return a + b

    with pytest.raises(
        ValueError,
        match="The provided `pipefuncs` should have only one leaf node, not 2.",
    ):
        NestedPipeFunc([f, g])


def test_nested_pipefunc_variant_different_output_name() -> None:
    @pipefunc(output_name="sum_", variant_group="op", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="diff", variant_group="op", variant="sub")
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
        variant_group="op",
        variant="add",
        output_name="double",
    )
    nf2 = NestedPipeFunc(
        [vp_sub.functions[0], vp_sub.functions[2]],
        variant_group="op",
        variant="sub",
        output_name="half",
    )

    vp2 = VariantPipeline([nf, nf2])
    pipeline1 = vp2.with_variant({"op": "add"})
    assert isinstance(pipeline1, Pipeline)
    assert pipeline1(a=1, b=2) == 6
    pipeline2 = vp2.with_variant({"op": "sub"})
    assert isinstance(pipeline2, Pipeline)
    assert pipeline2(a=1, b=2) == -0.5
