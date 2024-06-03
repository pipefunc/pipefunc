from __future__ import annotations

import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc


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

    assert str(f_c.mapspec) == "a[i], b[i] -> c[i]"
    assert str(f_d.mapspec) == "c[i], b[i], x[j] -> d[i, j]"
    assert str(f_e.mapspec) == "d[i, j], c[i], x[j] -> e[i, j]"

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

    assert str(f_c.mapspec) == "a[i], b[j] -> c[i, j]"
    assert str(f_d.mapspec) == "c[i, j], b[j], x[k] -> d[i, j, k]"
    assert str(f_e.mapspec) == "d[i, j, k], c[i, j], x[k] -> e[i, j, k]"

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

    assert str(func.mapspec) == "a[i, k], b[j, l] -> result[i, j, k, l]"


def test_add_mapspec_axis_parameter_in_output() -> None:
    @pipefunc(output_name="result", mapspec="a[i, j] -> result[i, j]")
    def func(a):
        return a

    pipeline = Pipeline([func])

    pipeline.add_mapspec_axis("a", axis="k")

    assert str(func.mapspec) == "a[i, j, k] -> result[i, j, k]"


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

    assert str(func1.mapspec) == "a[i, l], b[j] -> out1[i, j, l], out2[i, j, l]"
    assert str(func2.mapspec) == "out1[i, j, l], c[k] -> out3[i, j, k, l]"
    assert str(func3.mapspec) == "out3[:, :, :, l], out2[:, :, l] -> out4[l]"


def test_multiple_outputs_order() -> None:
    @pipefunc(output_name=("out1", "out2"), mapspec="a[i] -> out2[i], out1[i]")
    def func(a):
        return a, a + 1

    with pytest.raises(
        ValueError,
        match="does not match the output_names in the MapSpec",
    ):
        Pipeline([func])
