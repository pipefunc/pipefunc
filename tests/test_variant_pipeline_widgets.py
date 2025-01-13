from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest

from pipefunc import VariantPipeline, pipefunc

has_graphviz = importlib.util.find_spec("graphviz") is not None
has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None
has_rich = importlib.util.find_spec("rich") is not None
has_psutil = importlib.util.find_spec("psutil") is not None
has_matplotlib = importlib.util.find_spec("matplotlib") is not None


@pytest.fixture(autouse=True)
def patched_show():
    if not has_matplotlib:
        yield
        return
    import matplotlib.pyplot as plt

    with patch.object(plt, "show") as mock_show:
        yield mock_show


@pytest.mark.skipif(not has_graphviz, reason="requires graphviz")
def test_visualize_default_backend() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        return a - b

    pipeline = VariantPipeline([f, f_alt], default_variant="add")
    # Assuming the default backend is 'graphviz_widget'
    result = pipeline.visualize()
    if has_ipywidgets and has_graphviz:
        assert type(result).__name__ == "VBox"
    else:
        assert result is None


@pytest.mark.skipif(not has_graphviz, reason="requires graphviz")
def test_visualize_graphviz_backend() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        return a - b

    pipeline = VariantPipeline([f, f_alt], default_variant="add")
    result = pipeline.visualize(backend="graphviz")
    assert type(result).__name__ == "VBox"


@pytest.mark.skipif(
    not has_ipywidgets or not has_graphviz,
    reason="requires ipywidgets and graphviz",
)
def test_visualize_graphviz_widget_backend() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        return a - b

    pipeline = VariantPipeline([f, f_alt], default_variant="add")
    result = pipeline.visualize(backend="graphviz_widget")
    assert type(result).__name__ == "VBox"


@pytest.mark.skipif(not has_matplotlib, reason="requires matplotlib")
def test_visualize_matplotlib_backend() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        return a - b

    pipeline = VariantPipeline([f, f_alt], default_variant="add")
    result = pipeline.visualize(backend="matplotlib")
    assert type(result).__name__ == "VBox"
    assert type(result.children[-1]).__name__ == "Output"


@pytest.mark.skipif(
    not has_ipywidgets or not has_graphviz,
    reason="requires ipywidgets and graphviz",
)
def test_visualize_with_variant_selection() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        return a - b

    pipeline = VariantPipeline([f, f_alt], default_variant="add")
    result = pipeline.with_variant(select={"op1": "sub"}).visualize()
    assert type(result).__name__ == "Digraph"


@pytest.mark.skipif(not has_ipywidgets or not has_rich, reason="requires ipywidgets and rich")
def test_repr_mimebundle_with_ipywidgets_and_rich() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    pipeline = VariantPipeline([f], default_variant="add")
    result = pipeline._repr_mimebundle_()
    assert "text/plain" in result


def test_repr_mimebundle_without_ipywidgets_or_rich() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    pipeline = VariantPipeline([f], default_variant="add")
    result = pipeline._repr_mimebundle_()
    assert "text/plain" in result
