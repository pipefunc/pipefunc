"""Tests for pipefunc.py."""

from __future__ import annotations

from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from pipefunc import Pipeline, pipefunc


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
    def g(b, c, x=1):
        return b, c, x

    pipeline = Pipeline([f, g])
    pipeline.visualize()
