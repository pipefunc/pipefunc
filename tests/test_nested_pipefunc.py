import pytest

from pipefunc import NestedPipeFunc, VariantPipeline, pipefunc
from pipefunc.map import MapSpec
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
    assert nf(a=1) == 3
    nf.update_defaults({"a": 5, "b": 10})
    assert nf.defaults == {"a": 5, "b": 10}
    assert nf() == 15

    # Test with multiple outputs and defaults
    @pipefunc(output_name=("e", "f"))
    def h(x, y=10):
        return x, y

    nf2 = NestedPipeFunc([h], output_name=("e", "f"))
    assert nf2.defaults == {"y": 10}
    assert nf2(x=5) == (5, 10)


def test_nested_pipefunc_bound() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c

    nf = NestedPipeFunc([f, g])
    nf.update_bound({"a": 1})
    assert nf.bound == {"a": 1}
    assert nf(a=10, b=2) == 3  # a is bound to 1, so input a=10 is ignored
    nf.update_bound({"b": 5})
    assert nf.bound == {"a": 1, "b": 5}
    assert nf(a=100, b=200) == 6  # a and b are bound to 1 and 5 respectively

    # Test with multiple outputs
    @pipefunc(output_name=("e", "f"))
    def h(x, y):
        return x, y

    nf2 = NestedPipeFunc([h], output_name=("e", "f"))
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
    assert nf.mapspec.output_indices == {"i"}
    assert nf.mapspec.inputs == (("a", "i"), ("b", "i"))
    assert nf.mapspec.outputs == (("d", "i"),)

    # Test with different mapspecs
    @pipefunc(output_name="e", mapspec="x[i], y[j] -> e[i, j]")
    def h(x, y):
        return x * y

    @pipefunc(output_name="f", mapspec="e[i, j] -> f[i]")
    def k(e):
        return sum(e)

    nf2 = NestedPipeFunc([h, k])
    assert nf2.mapspec is not None
    assert nf2.mapspec.input_indices == {"i", "j"}
    assert nf2.mapspec.output_indices == {"i"}
    assert nf2.mapspec.inputs == (("x", "i"), ("y", "j"))
    assert nf2.mapspec.outputs == (("f", "i"),)

    # Test with custom mapspec
    nf3 = NestedPipeFunc(
        [f, g],
        mapspec=MapSpec(
            inputs=(("a", "i"), ("b", "i")),
            outputs=(("d", "i"),),
        ),
    )
    assert nf3.mapspec is not None
    assert nf3.mapspec.input_indices == {"i"}
    assert nf3.mapspec.output_indices == {"i"}
    assert nf3.mapspec.inputs == (("a", "i"), ("b", "i"))
    assert nf3.mapspec.outputs == (("d", "i"),)


def test_nested_pipefunc_resources() -> None:
    @pipefunc(output_name="c", resources=Resources(cpus=2, memory="1GB"))
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", resources=Resources(cpus=1, memory="2GB"))
    def g(c):
        return c

    nf = NestedPipeFunc([f, g])
    assert nf.resources is not None
    assert nf.resources.cpus == 2
    assert nf.resources.memory == "2GB"

    # Test with resources specified in NestedPipeFunc
    nf2 = NestedPipeFunc(
        [f, g],
        resources=Resources(cpus=4, memory="4GB"),
    )
    assert nf2.resources is not None
    assert nf2.resources.cpus == 4
    assert nf2.resources.memory == "4GB"

    # Test combining different resources
    @pipefunc(output_name="e", resources=Resources(gpus=1))
    def h(x):
        return x

    nf3 = NestedPipeFunc([f, g, h])
    assert nf3.resources is not None
    assert nf3.resources.cpus == 2
    assert nf3.resources.memory == "2GB"
    assert nf3.resources.gpus == 1

    # Test with no resources
    @pipefunc(output_name="i")
    def k(x):
        return x

    nf4 = NestedPipeFunc([k])
    assert nf4.resources is None


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
    assert vp.with_variant("add")(a=1, b=2) == 3
    assert vp.with_variant("sub")(a=1, b=2) == -1


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
    assert vp.with_variant({"op_mult": "add_yes"})(a=1, b=2) == 6
    assert vp.with_variant({"op_mult": "sub_no"})(a=1, b=2) == -1


def test_nested_pipefunc_with_scope() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c

    nf = NestedPipeFunc([f, g])
    nf.update_scope("my_scope", "*", "*")
    assert nf.parameters == ("my_scope.a", "my_scope.b")
    assert nf.output_name == "my_scope.d"
    assert nf(my_scope={"a": 1, "b": 2}) == 3


def test_nested_pipefunc_with_post_execution_hook() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c

    hook_calls = []

    def my_hook(func, result, kwargs):
        hook_calls.append((func.func.__name__, result, kwargs))

    nf = NestedPipeFunc([f, g], post_execution_hook=my_hook)
    assert nf(a=1, b=2) == 3
    assert len(hook_calls) == 1
    assert hook_calls[0][0] == nf.func.__name__
    assert hook_calls[0][1] == 3
    assert hook_calls[0][2] == {"a": 1, "b": 2}


def test_nested_pipefunc_internal_shape() -> None:
    @pipefunc(output_name="c", mapspec="a[i] -> c[i]")
    def f(a):
        return a

    @pipefunc(output_name="d", mapspec="c[i] -> d[i]")
    def g(c):
        return c

    nf = NestedPipeFunc([f, g])
    # You can't set internal shape on a NestedPipeFunc
    with pytest.raises(AttributeError):
        nf.internal_shape = "?"


def test_nested_pipefunc_output_picker() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c

    nf = NestedPipeFunc([f, g])
    # You can't set output picker on a NestedPipeFunc
    with pytest.raises(AttributeError):
        nf.output_picker

    @pipefunc(output_name=("c", "d"), output_picker=lambda x, y: x[0] if y == "c" else x[1])
    def h(a, b):
        return a + b, a * b

    @pipefunc(output_name="e")
    def i(c, d):
        return c + d

    nf = NestedPipeFunc([h, i], output_name="e")
    assert nf(a=1, b=2) == 5

    nf = NestedPipeFunc(
        [h, i],
        output_name="e",
        output_picker=lambda x, y: x[0] if y == "e" else x[1],
    )
    assert nf(a=1, b=2) == 5

    nf = NestedPipeFunc(
        [h, i],
        output_name=("e", "c"),
        output_picker=lambda x, y: x[0] if y == "e" else x[1],
    )
    assert nf(a=1, b=2) == (5, 3)


def test_nested_pipefunc_error_snapshot() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        raise ValueError("Intentional error")

    nf = NestedPipeFunc([f, g])
    with pytest.raises(ValueError, match="Intentional error"):
        nf(a=1, b=2)
    assert nf.error_snapshot is not None
    assert isinstance(nf.error_snapshot, ErrorSnapshot)
    assert nf.error_snapshot.args == ()
    assert nf.error_snapshot.kwargs == {"a": 1, "b": 2}


def test_nested_pipefunc_debug() -> None:
    @pipefunc(output_name="c", debug=True)
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(c):
        return c

    nf = NestedPipeFunc([f, g], debug=True)
    assert nf.debug is False


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
    @pipefunc(output_name="sum", variant_group="op", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="diff", variant_group="op", variant="sub")
    def f2(a, b):
        return a - b

    @pipefunc(output_name="double")
    def g(sum):
        return sum * 2

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
    assert vp2.with_variant({"op": "add"})(a=1, b=2) == 6
    assert vp2.with_variant({"op": "sub"})(a=1, b=2) == -0.5
