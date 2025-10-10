from __future__ import annotations

import importlib.util
from concurrent.futures import Executor
from typing import Literal

import numpy as np
import pytest

from pipefunc import NestedPipeFunc, PipeFunc, Pipeline, pipefunc
from pipefunc._pipeline._mapspec import replace_none_in_axes
from pipefunc.map._mapspec import MapSpec
from pipefunc.typing import Array  # noqa: TC001

has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None


def test_adding_zipped_axes_to_mapspec_less_pipeline() -> None:
    @pipefunc(output_name="c")
    def f_c(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f_d(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f_e(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f_c, f_d, f_e])
    pipeline.add_mapspec_axis("a", axis="i")
    pipeline.add_mapspec_axis("b", axis="i")
    pipeline.add_mapspec_axis("x", axis="j")

    assert str(pipeline["c"].mapspec) == "a[i], b[i] -> c[i]"
    assert str(pipeline["d"].mapspec) == "c[i], b[i], x[j] -> d[i, j]"
    assert str(pipeline["e"].mapspec) == "d[i, j], c[i], x[j] -> e[i, j]"

    assert pipeline.mapspecs_as_strings == [
        "a[i], b[i] -> c[i]",
        "c[i], b[i], x[j] -> d[i, j]",
        "d[i, j], c[i], x[j] -> e[i, j]",
    ]
    axes = pipeline.mapspec_axes
    assert axes == {
        "a": ("i",),
        "b": ("i",),
        "c": ("i",),
        "x": ("j",),
        "d": ("i", "j"),
        "e": ("i", "j"),
    }
    dimensions = pipeline.mapspec_dimensions
    assert dimensions.keys() == axes.keys()
    assert all(dimensions[k] == len(v) for k, v in axes.items())


def test_adding_axes_to_mapspec_less_pipeline() -> None:
    @pipefunc(output_name="c")
    def f_c(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f_d(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f_e(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f_c, f_d, f_e])
    pipeline.add_mapspec_axis("a", axis="i")
    pipeline.add_mapspec_axis("b", axis="j")
    pipeline.add_mapspec_axis("x", axis="k")

    assert str(pipeline["c"].mapspec) == "a[i], b[j] -> c[i, j]"
    assert str(pipeline["d"].mapspec) == "c[i, j], b[j], x[k] -> d[i, j, k]"
    assert str(pipeline["e"].mapspec) == "d[i, j, k], c[i, j], x[k] -> e[i, j, k]"

    assert pipeline.mapspecs_as_strings == [
        "a[i], b[j] -> c[i, j]",
        "c[i, j], b[j], x[k] -> d[i, j, k]",
        "d[i, j, k], c[i, j], x[k] -> e[i, j, k]",
    ]


def test_add_mapspec_axis_multiple_axes() -> None:
    @pipefunc(output_name="result", mapspec="a[i], b[j] -> result[i, j]")
    def func(a, b):
        return a + b

    pipeline = Pipeline([func])

    pipeline.add_mapspec_axis("a", axis="k")
    pipeline.add_mapspec_axis("b", axis="l")

    assert str(pipeline["result"].mapspec) == "a[i, k], b[j, l] -> result[i, j, k, l]"


def test_add_mapspec_axis_parameter_in_output() -> None:
    @pipefunc(output_name="result", mapspec="a[i, j] -> result[i, j]")
    def func(a):
        return a

    pipeline = Pipeline([func])

    pipeline.add_mapspec_axis("a", axis="k")

    assert str(pipeline["result"].mapspec) == "a[i, j, k] -> result[i, j, k]"


def test_consistent_indices() -> None:
    with pytest.raises(
        ValueError,
        match="All axes should have the same name at the same index",
    ):
        Pipeline(
            [
                PipeFunc(lambda a, b: a + b, "f", mapspec="a[i], b[i] -> f[i]"),
                PipeFunc(lambda f, g: f + g, "h", mapspec="f[k], g[k] -> h[k]"),
            ],
        )

    with pytest.raises(
        ValueError,
        match="All axes should have the same length",
    ):
        Pipeline(
            [
                PipeFunc(lambda a: a, "f", mapspec="a[i] -> f[i]"),
                PipeFunc(lambda a: a, "g", mapspec="a[i, j] -> g[i, j]"),
            ],
        )


def test_consistent_indices_multiple_functions() -> None:
    pipeline = Pipeline(
        [
            PipeFunc(lambda a, b: a + b, "f", mapspec="a[i], b[j] -> f[i, j]"),
            PipeFunc(lambda f, c: f * c, "g", mapspec="f[i, j], c[k] -> g[i, j, k]"),
            PipeFunc(lambda g, d: g + d, "h", mapspec="g[i, j, k], d[l] -> h[i, j, k, l]"),
        ],
    )
    pipeline._validate_mapspec()  # Should not raise any error


def test_validate_mapspec() -> None:
    def f(x: int) -> int:
        return x

    with pytest.raises(
        ValueError,
        match="The input of the function `f` should match the input of the MapSpec",
    ):
        PipeFunc(
            f,
            output_name="y",
            mapspec="x[i], yolo[i] -> y[i]",
        )

    with pytest.raises(
        ValueError,
        match="The output of the function `f` should match the output of the MapSpec",
    ):
        PipeFunc(
            f,
            output_name="y",
            mapspec="x[i] -> yolo[i]",
        )


def test_add_mapspec_axis_unused_parameter() -> None:
    @pipefunc(output_name="result", mapspec="a[i] -> result[i]")
    def func(a):
        return a

    pipeline = Pipeline([func])

    pipeline.add_mapspec_axis("unused_param", axis="j")

    assert str(func.mapspec) == "a[i] -> result[i]"


def test_add_mapspec_axis_complex_pipeline() -> None:
    @pipefunc(output_name=("out1", "out2"), mapspec="a[i], b[j] -> out1[i, j], out2[i, j]")
    def func1(a, b):
        return a + b, a - b

    @pipefunc(output_name="out3", mapspec="out1[i, j], c[k] -> out3[i, j, k]")
    def func2(out1, c):
        return out1 * c

    @pipefunc(output_name="out4")
    def func3(out2, out3):
        return out2 + out3

    pipeline = Pipeline([func1, func2, func3])

    pipeline.add_mapspec_axis("a", axis="l")

    assert str(pipeline["out1"].mapspec) == "a[i, l], b[j] -> out1[i, j, l], out2[i, j, l]"
    assert str(pipeline["out3"].mapspec) == "out1[i, j, l], c[k] -> out3[i, j, k, l]"
    assert str(pipeline["out4"].mapspec) == "out3[:, :, :, l], out2[:, :, l] -> out4[l]"


def test_multiple_outputs_order() -> None:
    @pipefunc(output_name=("out1", "out2"), mapspec="a[i] -> out2[i], out1[i]")
    def func(a):
        return a, a + 1

    with pytest.raises(
        ValueError,
        match="does not match the output_names in the MapSpec",
    ):
        Pipeline([func])


def test_combining_mapspecs() -> None:
    @pipefunc(
        output_name="electrostatics",
        mapspec="'V_left[i], V_right[j], mesh[a, b, c], materials[a, b] -> electrostatics[i, j, a, b, c]'",
    )
    def electrostatics(V_left, V_right, mesh, materials):  # noqa: N803
        return 1

    @pipefunc(
        output_name="charge",
        mapspec="'electrostatics[i, j, a, b, c] -> charge[i, j, a, b, c]'",
    )
    def charge(electrostatics):
        return electrostatics

    expected = "V_left[i], V_right[j], materials[a, b], mesh[a, b, c] -> charge[i, j, a, b, c]"
    nf = NestedPipeFunc([electrostatics, charge], output_name="charge")
    assert str(nf.mapspec) == expected
    nf = NestedPipeFunc([electrostatics, charge])
    assert (
        str(nf.mapspec)
        == "V_left[i], V_right[j], materials[a, b], mesh[a, b, c] -> charge[i, j, a, b, c], electrostatics[i, j, a, b, c]"
    )

    pipeline = Pipeline([electrostatics, charge])
    pipeline.nest_funcs({"electrostatics", "charge"}, new_output_name="charge")
    assert pipeline.mapspecs_as_strings == [expected]
    assert len(pipeline.functions) == 1


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

    nf = NestedPipeFunc([f, g])
    assert nf.output_name == ("c", "d", "z")
    assert nf(a=1, b=2, x=3, y=4) == (3, 1, 7)


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


def test_mapping_over_default() -> None:
    @pipefunc(output_name="out", mapspec="a[i], b[i] -> out[i]", defaults={"b": [1, 2, 3]})
    def f(a, b):
        return a + b

    pipeline = Pipeline([f])
    r_map = pipeline.map(inputs={"a": [1, 2, 3]}, parallel=False, storage="dict")
    assert r_map["out"].output.tolist() == [2, 4, 6]


@pytest.mark.parametrize("dim", [3, "?"])
def test_calling_add_with_autogen_mapspec(dim: int | Literal["?"]):
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
        internal_shapes={"foo_out": (dim,)},
        parallel=False,
        storage="dict",
    )
    assert results["bar_out"].output.tolist() == [1, 4, 9]


def test_validation_parallel():
    pipeline = Pipeline([])
    with pytest.raises(
        ValueError,
        match="Cannot use an executor without `parallel=True`",
    ):
        pipeline.map({}, executor=Executor(), parallel=False, storage="dict")


@pytest.mark.skipif(not has_ipywidgets, reason="ipywidgets not installed")
def test_with_progress() -> None:
    @pipefunc(output_name="out", mapspec="a[i] -> out[i]")
    def f(a: int) -> int:
        return a

    @pipefunc(output_name="out_sum")
    def g(out: Array[int]) -> int:
        return sum(out)

    pipeline = Pipeline([f, g])
    r_map = pipeline.map(
        inputs={"a": [1, 2, 3]},
        show_progress="ipywidgets",
        parallel=True,
        storage="dict",
    )
    assert r_map["out"].output.tolist() == [1, 2, 3]
    assert r_map["out_sum"].output == 6

    r_map_sequential = pipeline.map(
        inputs={"a": [1, 2, 3]},
        show_progress="ipywidgets",
        parallel=False,
        storage="dict",
    )
    assert r_map_sequential["out"].output.tolist() == [1, 2, 3]
    assert r_map_sequential["out_sum"].output == 6


def test_replace_none_in_axes() -> None:
    """Test `replace_none_in_axes`.

    The main reason to add this test is to leave this comment here, ensuring I
    won't needlessly try to debugging this potentially confusing behavior.

    The fact below `"a[i, j]"` appears in the first mapspec output and that the
    `replace_none_in_axes` replaces the `None` in the second mapspec input with
    `"unnamed_0"` is not a problem! This is because no mapspec will be autogenerated
    with this axis.

    For example, pipeline.mapspec_axes will properly include `a[i, j]`.
    Also see `test_double_output_then_iterate_over_single_axis`.
    """
    mapspecs_as_strings = [
        "x[i], y[j] -> a[i, j]",
        "a[:, j] -> c[j]",
    ]
    mapspecs = [MapSpec.from_string(s) for s in mapspecs_as_strings]
    non_root_inputs = {"a": [None, "j"]}
    replace_none_in_axes(mapspecs, non_root_inputs, {})  # type: ignore[arg-type]
    assert non_root_inputs == {"a": ["unnamed_0", "j"]}


@pytest.mark.parametrize("show_progress", [True, False])
def test_zero_sizes_list_with_progress_bar(show_progress: bool) -> None:
    if show_progress and not has_ipywidgets:
        pytest.skip("ipywidgets not installed")

    @pipefunc(output_name="x")
    def generate_ints() -> list[int]:
        return []

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline_sum = Pipeline([generate_ints, double_it])
    results = pipeline_sum.map({}, show_progress=show_progress, parallel=False, storage="dict")
    assert results["y"].output.tolist() == []


def test_add_nd_mapspec_axis() -> None:
    @pipefunc(output_name="x")
    def f(a):
        return a

    pipeline = Pipeline([f])
    pipeline.add_mapspec_axis("a", axis="i, j")
    assert str(pipeline["x"].mapspec) == "a[i, j] -> x[i, j]"


def test_add_nd_mapspec_axis_to_existing_1d() -> None:
    """Test adding a 2D axis 'k, l' to input 'a' which already has axis 'i'."""

    @pipefunc(output_name="result", mapspec="a[i] -> result[i]")
    def func(a):
        return a

    pipeline = Pipeline([func])
    pipeline.add_mapspec_axis("a", axis="k, l")
    assert str(pipeline["result"].mapspec) == "a[i, k, l] -> result[i, k, l]"


def test_add_nd_mapspec_axis_to_existing_mixed() -> None:
    """Test adding a 2D axis 'k, l' to input 'b' which has 'j', while 'a' has 'i'."""

    @pipefunc(output_name="result", mapspec="a[i], b[j] -> result[i, j]")
    def func(a, b):
        return a + b

    pipeline = Pipeline([func])
    pipeline.add_mapspec_axis("b", axis="k, l")
    assert str(pipeline["result"].mapspec) == "a[i], b[j, k, l] -> result[i, j, k, l]"


def test_add_nd_mapspec_axis_to_existing_nd() -> None:
    """Test adding a 2D axis 'k, l' to input 'a' which already has 'i, j'."""

    @pipefunc(output_name="result", mapspec="a[i, j] -> result[i, j]")
    def func(a):
        return a

    pipeline = Pipeline([func])
    pipeline.add_mapspec_axis("a", axis="k, l")
    assert str(pipeline["result"].mapspec) == "a[i, j, k, l] -> result[i, j, k, l]"


def test_add_nd_mapspec_axis_propagates() -> None:
    """Test adding 2D axis 'k, l' to 'a' propagates through 'c'."""

    @pipefunc(output_name="c", mapspec="a[i] -> c[i]")
    def f_c(a):
        return a

    @pipefunc(output_name="d", mapspec="c[i], b[j] -> d[i, j]")
    def f_d(c, b):
        return c + b

    pipeline = Pipeline([f_c, f_d])
    pipeline.add_mapspec_axis("a", axis="k, l")

    assert str(pipeline["c"].mapspec) == "a[i, k, l] -> c[i, k, l]"
    assert str(pipeline["d"].mapspec) == "c[i, k, l], b[j] -> d[i, j, k, l]"


def test_add_nd_mapspec_axis_fan_in() -> None:
    """Test adding 2D axis 'k, l' to 'a' in a fan-in structure."""

    @pipefunc(output_name="c", mapspec="a[i] -> c[i]")
    def f_c(a):
        return a

    @pipefunc(output_name="d", mapspec="b[j] -> d[j]")
    def f_d(b):
        return b

    @pipefunc(output_name="e", mapspec="c[i], d[j] -> e[i, j]")
    def f_e(c, d):
        return c + d

    pipeline = Pipeline([f_c, f_d, f_e])
    pipeline.add_mapspec_axis("a", axis="k, l")

    assert str(pipeline["c"].mapspec) == "a[i, k, l] -> c[i, k, l]"
    assert str(pipeline["d"].mapspec) == "b[j] -> d[j]"  # Unchanged
    assert str(pipeline["e"].mapspec) == "c[i, k, l], d[j] -> e[i, j, k, l]"


def test_add_nd_mapspec_axis_fan_out() -> None:
    """Test adding 2D axis 'k, l' to 'a' in a fan-out structure."""

    @pipefunc(output_name="c", mapspec="... -> c[i]")
    def f_c(a):
        return [a, a, a]

    @pipefunc(output_name="d", mapspec="b[j] -> d[j]")
    def f_d(b):
        return b

    @pipefunc(output_name="e", mapspec="c[i], d[j] -> e[i, j]")
    def f_e(c, d):
        return c + d

    pipeline = Pipeline([f_c, f_d, f_e])
    pipeline.add_mapspec_axis("a", axis="k, l")

    assert str(pipeline["c"].mapspec) == "a[k, l] -> c[i, k, l]"
    assert str(pipeline["d"].mapspec) == "b[j] -> d[j]"  # Unchanged
    assert str(pipeline["e"].mapspec) == "c[i, k, l], d[j] -> e[i, j, k, l]"
    results = pipeline.map({"a": np.array([[1]]), "b": [1, 2]})
    assert results["e"].output.tolist() == [[[[2]], [[3]]], [[[2]], [[3]]], [[[2]], [[3]]]]
    assert results["e"].output.shape == (3, 2, 1, 1)


@pytest.fixture
def pipeline_complex() -> Pipeline:
    """A complex pipeline for testing."""

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

    return Pipeline([f1, f2, f3, f4, f5, f6, f7])  # type: ignore[list-item]


def test_update_mapspec_axes(pipeline_complex: Pipeline) -> None:
    pipeline = pipeline_complex.copy()
    pipeline.add_mapspec_axis("a", axis="i")
    pipeline.add_mapspec_axis("b", axis="j")
    assert pipeline.mapspec_axes["f1"] == ("i", "j")
    pipeline.update_mapspec_axes({"i": "ii", "j": "jj"})
    assert pipeline.mapspec_axes["f1"] == ("ii", "jj")
    assert "a[ii], b[jj] -> f1[ii, jj]" in pipeline.mapspecs_as_strings


def test_update_mapspec_axes_unknown_axis(pipeline_complex: Pipeline) -> None:
    pipeline = pipeline_complex.copy()
    pipeline.add_mapspec_axis("a", axis="i")
    with pytest.raises(ValueError, match="Unknown axes to rename"):
        pipeline.update_mapspec_axes({"z": "zz"})


def test_update_mapspec_axes_no_mapspec() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    pipeline = Pipeline([f])
    with pytest.raises(ValueError, match="Unknown axes to rename"):
        pipeline.update_mapspec_axes({"i": "ii"})


def test_update_mapspec_skip_mapspecless_function(pipeline_complex: Pipeline) -> None:
    pipeline = pipeline_complex.copy()
    pipeline.add_mapspec_axis("e", axis="i")
    # Skip mapspecless function f1
    pipeline.update_mapspec_axes({"i": "ii"})

    assert pipeline.mapspec_axes["f2"] == ("ii",)
    assert pipeline.mapspecs_as_strings == ["e[ii] -> f2[ii]", "f2[ii] -> f7[ii]"]
