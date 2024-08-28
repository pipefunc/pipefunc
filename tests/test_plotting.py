"""Tests for pipefunc.py."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from pipefunc import Pipeline, pipefunc

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def patched_show():
    with patch.object(plt, "show") as mock_show:
        yield mock_show


def test_plot():
    @pipefunc("c")
    def a(b):
        return b

    @pipefunc("d")
    def c(c):
        return c

    pipeline = Pipeline([a, c])
    pipeline.visualize()


def test_plot_with_defaults():
    @pipefunc("c")
    def f(a, b, x):
        return a, b, x

    @pipefunc("d")
    def g(b, c, x=1):
        return b, c, x

    pipeline = Pipeline([f, g])
    pipeline.visualize()


def test_plot_with_defaults_and_bound():
    @pipefunc("c", bound={"x": 2})
    def f(a, b, x):
        return a, b, x

    @pipefunc("d")
    def g(b, c, x="1" * 100):  # x is a long string that should be trimmed
        return b, c, x

    pipeline = Pipeline([f, g])
    pipeline.visualize(color_combinable=True)


def test_plot_with_mapspec(tmp_path: Path):
    @pipefunc("c", mapspec="a[i] -> c[i]")
    def f(a, b, x):
        return a, b, x

    @pipefunc("d", mapspec="b[i], c[i] -> d[i]")
    def g(b, c, x):
        return b, c, x

    pipeline = Pipeline([f, g])
    filename = tmp_path / "pipeline.png"
    pipeline.visualize(filename=filename)
    assert filename.exists()
    pipeline.visualize_holoviews()


def test_plot_nested_func():
    @pipefunc("c", bound={"x": 2})
    def f(a, b, x):
        return a, b, x

    @pipefunc("d")
    def g(b, c, x="1" * 100):  # x is a long string that should be trimmed
        return b, c, x

    pipeline = Pipeline([f, g])
    pipeline.nest_funcs("*")
    pipeline.visualize()


def test_plotting_resources():
    @pipefunc(output_name="c", resources_variable="resources", resources={"gpus": 8})
    def f_c(a, b, resources):
        return resources.gpus

    pipeline = Pipeline([f_c])
    pipeline.visualize()
