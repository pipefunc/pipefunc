"""Tests for pipefunc.py."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from pipefunc import NestedPipeFunc, Pipeline, pipefunc

if TYPE_CHECKING:
    from pathlib import Path

# Check for required libraries
import importlib

has_matplotlib = importlib.util.find_spec("matplotlib") is not None
has_holoviews = importlib.util.find_spec("holoviews") is not None
has_pygraphviz = importlib.util.find_spec("pygraphviz") is not None


@pytest.fixture(autouse=True)
def patched_show():
    if not has_matplotlib:
        yield
        return
    import matplotlib.pyplot as plt

    with patch.object(plt, "show") as mock_show:
        yield mock_show


@pytest.mark.skipif(not has_pygraphviz, reason="pygraphviz not installed")
def test_plot() -> None:
    import graphviz

    @pipefunc("c")
    def a(b):
        return b

    @pipefunc("d")
    def c(c):
        return c

    pipeline = Pipeline([a, c])
    fig = pipeline.visualize()
    assert isinstance(fig, graphviz.Digraph)


@pytest.mark.parametrize("backend", ["matplotlib", "holoviews", "graphviz"])
def test_plot_with_defaults(backend) -> None:
    @pipefunc("c")
    def f(a, b, x):
        return a, b, x

    @pipefunc("d")
    def g(b, c, x=1):
        return b, c, x

    pipeline = Pipeline([f, g])

    if backend == "matplotlib" and not has_matplotlib:
        pytest.skip("matplotlib not installed")
    elif backend == "holoviews" and not has_holoviews:
        pytest.skip("holoviews not installed")
    elif backend == "graphviz" and not has_pygraphviz:
        pytest.skip("pygraphviz not installed")

    pipeline.visualize(backend=backend)


@pytest.mark.skipif(not has_matplotlib, reason="matplotlib not installed")
def test_plot_with_defaults_and_bound() -> None:
    @pipefunc("c", bound={"x": 2})
    def f(a, b, x):
        return a, b, x

    @pipefunc("d")
    def g(b, c, x="1" * 100):  # x is a long string that should be trimmed
        return b, c, x

    pipeline = Pipeline([f, g])
    pipeline.visualize_matplotlib(color_combinable=True)


@pytest.mark.parametrize("backend", ["matplotlib", "holoviews"])
def test_plot_with_mapspec(tmp_path: Path, backend) -> None:
    @pipefunc("c", mapspec="a[i] -> c[i]")
    def f(a, b, x):
        return a, b, x

    @pipefunc("d", mapspec="b[i], c[i] -> d[i]")
    def g(b, c, x):
        return b, c, x

    pipeline = Pipeline([f, g])

    if backend == "matplotlib":
        if not has_matplotlib:
            pytest.skip("matplotlib not installed")
        filename = tmp_path / "pipeline.png"
        pipeline.visualize_matplotlib(filename=filename)
        assert filename.exists()
    elif backend == "holoviews":
        if not has_holoviews:
            pytest.skip("holoviews not installed")
        pipeline.visualize_holoviews()


@pytest.mark.skipif(not has_matplotlib, reason="matplotlib not installed")
def test_plot_nested_func() -> None:
    @pipefunc("c", bound={"x": 2})
    def f(a, b, x):
        return a, b, x

    @pipefunc("d")
    def g(b, c, x="1" * 100):  # x is a long string that should be trimmed
        return b, c, x

    pipeline = Pipeline([f, g])
    pipeline.nest_funcs("*")
    pipeline.visualize(backend="matplotlib")


@pytest.mark.skipif(not has_matplotlib, reason="matplotlib not installed")
def test_plotting_resources() -> None:
    @pipefunc(output_name="c", resources_variable="resources", resources={"gpus": 8})
    def f_c(a, b, resources):
        return resources.gpus

    pipeline = Pipeline([f_c])
    pipeline.visualize_matplotlib(figsize=10)


@pytest.fixture
def everything_pipeline() -> Pipeline:
    @pipefunc(output_name="c")
    def f(a: int, b: int) -> int: ...  # type: ignore[empty-body]
    @pipefunc(output_name="d")
    def g(b: int, c: int, x: int = 1) -> int: ...  # type: ignore[empty-body]
    @pipefunc(
        output_name="e",
        bound={"x": 2},
        resources={"cpus": 1, "gpus": 1},
        resources_variable="resources",
        mapspec="c[i] -> e[i]",
    )
    def h(c: int, d: int, x: int = 1, *, resources) -> int: ...  # type: ignore[empty-body]
    @pipefunc(output_name="i1")
    def i1(a: int): ...
    @pipefunc(output_name="i2")
    def i2(i1: dict[str, int]): ...

    i = NestedPipeFunc([i1, i2], output_name="i2")
    return Pipeline([f, g, h, i])


@pytest.mark.parametrize("backend", ["matplotlib", "holoviews", "graphviz"])
def test_visualize_graphviz(
    backend,
    patched_show,
    everything_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    if backend == "matplotlib" and not has_matplotlib:
        pytest.skip("matplotlib not installed")
    elif backend == "holoviews" and not has_holoviews:
        pytest.skip("holoviews not installed")
    elif backend == "graphviz" and not has_pygraphviz:
        pytest.skip("pygraphviz not installed")

    everything_pipeline.visualize(backend=backend)
    if backend == "graphviz":
        everything_pipeline.visualize_graphviz(
            filename=tmp_path / "graphviz.svg",
            figsize=10,
            include_full_mapspec=True,
        )


@pytest.mark.skipif(not has_pygraphviz, reason="pygraphviz not installed")
def test_visualize_graphviz_with_typing():
    @pipefunc(output_name="c")
    def f(a: int, b: int) -> UnresolvableTypeHere:  # type: ignore[name-defined]  # noqa: F821
        return a + b

    @pipefunc(output_name="d")
    def g(b: int, c: int, x: int = 1) -> int:
        return b + c + x

    pipeline = Pipeline([f, g])
    pipeline.visualize_graphviz(return_type="html")
