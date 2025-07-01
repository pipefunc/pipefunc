"""Tests for pipefunc.py."""

from __future__ import annotations

import importlib
import shutil
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from pipefunc import NestedPipeFunc, Pipeline, pipefunc
from pipefunc._plotting import _AUTO_GROUP_ARGS_THRESHOLD

if TYPE_CHECKING:
    from pathlib import Path


has_matplotlib = importlib.util.find_spec("matplotlib") is not None
has_holoviews = importlib.util.find_spec("holoviews") is not None
has_graphviz = importlib.util.find_spec("graphviz") is not None
has_anywidget = importlib.util.find_spec("graphviz_anywidget") is not None
has_graphviz_exec = shutil.which("dot") is not None


@pytest.fixture(autouse=True)
def patch_pyviz_comms():
    # To avoid https://github.com/holoviz/pyviz_comms/issues/137
    # Remove once https://github.com/holoviz/pyviz_comms/pull/138 is merged
    try:
        with patch("pyviz_comms._in_ipython", new=False) as mock_in_ipython:
            yield mock_in_ipython
    except ImportError:
        yield


@pytest.fixture(autouse=True)
def patched_show():
    if not has_matplotlib:
        yield
        return
    import matplotlib.pyplot as plt

    with patch.object(plt, "show") as mock_show:
        yield mock_show


@pytest.mark.skipif(not has_graphviz, reason="graphviz not installed")
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
    elif backend == "graphviz" and not has_graphviz:
        pytest.skip("graphviz not installed")

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


@pytest.mark.parametrize("backend", ["matplotlib", "holoviews", "graphviz"])
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
    elif backend == "graphviz":
        if not has_graphviz:
            pytest.skip("graphviz not installed")
        pipeline.visualize_graphviz()


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
    def f(a: int, b: int, aa=1, bb=2, cc=3, dd=4) -> int: ...  # type: ignore[empty-body]
    @pipefunc(output_name="d", mapspec="... -> d[j]")
    def g(b: int, c: int, x: int = 1) -> int: ...  # type: ignore[empty-body]
    @pipefunc(
        output_name="e",
        bound={"x": 2},
        resources={"cpus": 1, "gpus": 1},
        resources_variable="resources",
        mapspec="c[i], d[j] -> e[i, j]",
    )
    def h(c: int, d: int, x: int = 1, *, resources) -> int: ...  # type: ignore[empty-body]
    @pipefunc(output_name="i1")
    def i1(a: int): ...
    @pipefunc(output_name="i2")
    def i2(i1: dict[str, int]): ...

    i = NestedPipeFunc([i1, i2], output_name="i2")
    return Pipeline([f, g, h, i])


@pytest.fixture
def scoped_pipeline() -> Pipeline:
    """Pipeline fixture with functions defined within scopes."""

    @pipefunc(output_name="c", scope="scope1")
    def f(a: int, b: int = 5) -> int: ...  # type: ignore[empty-body]
    @pipefunc(output_name="d", scope="scope1")
    def g(c: int) -> int: ...  # type: ignore[empty-body]
    @pipefunc(output_name="e", scope="scope2")
    def h(a: int, d: int = 10) -> int: ...  # type: ignore[empty-body]
    @pipefunc(output_name="f")  # No scope
    def i(e: int) -> int: ...  # type: ignore[empty-body]

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
    elif backend == "graphviz" and (not has_graphviz or not has_graphviz_exec):
        pytest.skip("graphviz not installed")

    everything_pipeline.visualize(backend=backend)
    if backend == "graphviz":
        from pipefunc._plotting import GraphvizStyle

        everything_pipeline.visualize_graphviz(
            filename=tmp_path / "graphviz.svg",
            figsize=10,
            include_full_mapspec=True,
            style=GraphvizStyle(background_color="transparent"),
        )


@pytest.mark.skipif(
    not has_anywidget or not has_graphviz,
    # NOTE: This should even work if 'dot' is not installed because it uses the Wasm graphviz
    reason="graphviz-anywidget not installed",
)
def test_plotting_widget(everything_pipeline: Pipeline) -> None:
    # Note: Not sure how to test this properly, just make sure it runs
    import ipywidgets

    widget = everything_pipeline.visualize(backend="graphviz_widget")
    extra_controls_hbox, second, *last = widget.children
    reset_button, freeze, direction_selector, search_input, search_type_selector, case_toggle = (
        second.children
    )
    reset_button.click()
    direction_selector.value = "downstream"
    search_input.value = "c"
    search_type_selector.value = "included"
    case_toggle.value = True

    # Check for hide defaults toggle (everything_pipeline has defaults)
    hide_defaults_toggle = next(
        (
            item
            for item in extra_controls_hbox.children
            if isinstance(item, ipywidgets.ToggleButton) and "default args" in item.description
        ),
        None,
    )
    assert hide_defaults_toggle is not None
    assert hide_defaults_toggle.value is False
    # Simulate click
    hide_defaults_toggle.value = True  # Trigger the observer
    assert hide_defaults_toggle.description == "Hide default args"
    assert hide_defaults_toggle.icon == "eye-slash"
    hide_defaults_toggle.value = False  # Trigger again
    assert hide_defaults_toggle.description == "Show default args"
    assert hide_defaults_toggle.icon == "eye"

    # Check that scope accordion is NOT present (everything_pipeline has no scopes)
    scope_accordion = next(
        (item for item in extra_controls_hbox.children if isinstance(item, ipywidgets.Accordion)),
        None,
    )
    assert scope_accordion is None

    # Check that orient dropdown is present
    orient_dropdown = next(
        (item for item in extra_controls_hbox.children if isinstance(item, ipywidgets.Dropdown)),
        None,
    )
    assert orient_dropdown is not None

    # Check that group args toggle is present
    assert len(everything_pipeline.graph.nodes) > _AUTO_GROUP_ARGS_THRESHOLD
    group_args_toggle = next(
        (
            item
            for item in extra_controls_hbox.children
            if isinstance(item, ipywidgets.ToggleButton) and "Ungroup args" in item.description
        ),
        None,
    )
    assert group_args_toggle is not None
    assert group_args_toggle.value is True
    assert group_args_toggle.description == "Ungroup args"
    assert group_args_toggle.icon == "minus-square-o"
    assert group_args_toggle.button_style == "success"
    group_args_toggle.value = False  # Simulate click
    assert group_args_toggle.description == "Group args"
    assert group_args_toggle.icon == "plus-square-o"
    assert group_args_toggle.button_style == "info"


@pytest.mark.skipif(
    not has_anywidget or not has_graphviz,
    reason="graphviz-anywidget not installed",
)
def test_plotting_widget_controls(scoped_pipeline: Pipeline) -> None:
    """Test the extra controls (scope collapse, hide defaults) on the widget."""
    # Note: Still hard to test the visual output or exact dot source changes easily.
    # Focus on widget creation and interaction without errors.
    import ipywidgets

    widget = scoped_pipeline.visualize(backend="graphviz_widget")
    assert isinstance(widget, ipywidgets.VBox)

    # Find controls
    extra_controls_hbox = widget.children[0]  # The HBox from _extra_controls_factory

    # Check for hide defaults toggle (scoped_pipeline has defaults)
    hide_defaults_toggle = next(
        (
            item
            for item in extra_controls_hbox.children
            if isinstance(item, ipywidgets.ToggleButton) and "default args" in item.description
        ),
        None,
    )
    assert hide_defaults_toggle is not None
    assert hide_defaults_toggle.value is False
    # Simulate click
    hide_defaults_toggle.value = True  # Trigger the observer
    assert hide_defaults_toggle.description == "Hide default args"
    assert hide_defaults_toggle.icon == "eye-slash"
    hide_defaults_toggle.value = False  # Trigger again
    assert hide_defaults_toggle.description == "Show default args"
    assert hide_defaults_toggle.icon == "eye"

    # Check for scope collapse accordion and toggles (scoped_pipeline has scopes)
    scope_accordion = next(
        (item for item in extra_controls_hbox.children if isinstance(item, ipywidgets.Accordion)),
        None,
    )
    assert scope_accordion is not None
    assert scope_accordion.get_title(0) == "Collapse Scopes"
    scope_vbox = scope_accordion.children[0]
    assert isinstance(scope_vbox, ipywidgets.VBox)
    scope_toggles = scope_vbox.children
    assert len(scope_toggles) == 2  # scope1, scope2
    assert {toggle.description for toggle in scope_toggles} == {"scope1", "scope2"}

    # Simulate clicking scope toggles
    scope1_toggle = next(t for t in scope_toggles if t.description == "scope1")
    scope2_toggle = next(t for t in scope_toggles if t.description == "scope2")

    assert scope1_toggle.value is False
    assert scope1_toggle.icon == "plus-square-o"
    scope1_toggle.value = True  # Trigger observer
    assert scope1_toggle.value is True
    assert scope1_toggle.icon == "minus-square-o"

    assert scope2_toggle.value is False
    assert scope2_toggle.icon == "plus-square-o"
    scope2_toggle.value = True  # Trigger observer
    assert scope2_toggle.value is True
    assert scope2_toggle.icon == "minus-square-o"

    # Collapse one scope
    scope1_toggle.value = False  # Trigger observer
    assert scope1_toggle.value is False
    assert scope1_toggle.icon == "plus-square-o"
    assert scope2_toggle.value is True  # Should remain collapsed
    assert scope2_toggle.icon == "minus-square-o"


@pytest.mark.skipif(not has_graphviz or not has_graphviz_exec, reason="graphviz not installed")
def test_visualize_graphviz_with_typing():
    @pipefunc(output_name="c")
    def f(a: int, b: int) -> UnresolvableTypeHere:  # type: ignore[name-defined]  # noqa: F821
        return a + b

    @pipefunc(output_name="d")
    def g(b: int, c: int, x: int = 1) -> int:
        return b + c + x

    with pytest.warns(UserWarning, match="Unresolvable type hint: `UnresolvableTypeHere`"):
        pipeline = Pipeline([f, g])
    pipeline.visualize_graphviz(return_type="html")


@pytest.mark.skipif(not has_graphviz or not has_graphviz_exec, reason="graphviz not installed")
def test_collapse_scope_plot_with_mapspecs():
    @pipefunc(output_name="c", mapspec="a[i] -> c[i]", scope="foo")
    def f(a: int, b: int) -> int:
        return a + b

    @pipefunc(output_name="d", mapspec="c[i] -> d[i]", scope="foo")
    def g(c: int) -> int:
        return c * 2

    @pipefunc(output_name="e", mapspec="foo.d[i] -> bar.e[i]", renames={"d": "foo.d", "e": "bar.e"})
    def h(d: int) -> int:
        return d * 3

    pipeline = Pipeline([f, g, h])
    results = pipeline.map(
        inputs={"foo.a": [1, 2], "foo.b": 3},
        parallel=False,
        storage="dict",
    )
    assert results["foo.c"].output.tolist() == [4, 5]
    assert results["foo.d"].output.tolist() == [8, 10]
    assert results["bar.e"].output.tolist() == [24, 30]

    # Test that both work
    pipeline.visualize_graphviz(collapse_scopes=True)
    pipeline.visualize_graphviz(collapse_scopes=False)


@pytest.mark.skipif(not has_graphviz or not has_graphviz_exec, reason="graphviz not installed")
@pytest.mark.parametrize("min_arg_group_size", [None, 1, 2, 3])
def test_min_arg_group_size(min_arg_group_size: int | None):
    @pipefunc(output_name="d", mapspec="foo.a[i] -> foo.d[i]", scope="foo")
    def f(a: int, b: int, c: int) -> int:
        return a + b + c

    @pipefunc(output_name="e")
    def g(d: int, x: int):
        return d + x

    pipeline = Pipeline([f, g])
    pipeline.visualize_graphviz(min_arg_group_size=min_arg_group_size)


@pytest.mark.skipif(not has_graphviz or not has_graphviz_exec, reason="graphviz not installed")
def test_min_arg_group_size_with_ungroupable():
    @pipefunc(output_name="d")
    def f(a: int) -> int:
        return a

    pipeline = Pipeline([f])
    pipeline.visualize_graphviz(min_arg_group_size=2)


@pytest.mark.skipif(not has_graphviz or not has_graphviz_exec, reason="graphviz not installed")
def test_hide_default_args():
    @pipefunc(output_name="d")
    def f(a: int, b: int = 1) -> int:
        return a + b

    pipeline = Pipeline([f])
    pipeline.visualize_graphviz(hide_default_args=True)
