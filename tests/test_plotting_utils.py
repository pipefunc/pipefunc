import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc._plotting_utils import _would_create_cycle, collapsed_scope_graph


def test_linear_pipeline_would_create_cycle() -> None:
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

    assert _would_create_cycle(pipeline.graph, [pipeline["x"], pipeline["z"]])
    assert not _would_create_cycle(pipeline.graph, [pipeline["x"], pipeline["y"]])
    assert not _would_create_cycle(pipeline.graph, [pipeline["x"]])
    assert not _would_create_cycle(pipeline.graph, [pipeline["y"], pipeline["z"]])
    assert not _would_create_cycle(pipeline.graph, [pipeline["x"], pipeline["y"], pipeline["z"]])
    new_graph = collapsed_scope_graph(pipeline.graph, ["scope"])
    assert len(pipeline.graph.nodes) == 3
    assert len(new_graph.nodes) == 3  # No change


def test_get_collapsed_scope_graph() -> None:
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
    pipeline.update_scope("scope", inputs="*", outputs="*")
    new_graph = collapsed_scope_graph(pipeline.graph, ["scope"])
    assert len(new_graph.nodes) == 1


def test_get_collapsed_scope_graph_each_scope_only_once() -> None:
    @pipefunc(output_name="foo.x")
    def f(a, b):
        return a + b

    @pipefunc(output_name="bar.y", renames={"x": "foo.x"})
    def g(x):
        return 2 * x

    pipeline = Pipeline([f, g])
    new_graph = collapsed_scope_graph(pipeline.graph, ["scope"])
    assert len(pipeline.graph.nodes) == 2 + 2  # 2 functions + 2 inputs
    assert len(new_graph.nodes) == 2 + 2  # no change


def test_nested_pipefunc_uncombinable_mapspecs() -> None:
    @pipefunc(output_name="c", mapspec="... -> c[i]")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", mapspec="c[i] -> d[i]")
    def g(c):
        return c * 2

    pipeline = Pipeline([f, g])
    pipeline.update_scope("foo", inputs={"a", "b"}, outputs={"c", "d"})

    graph = collapsed_scope_graph(pipeline.graph, scopes_to_collapse=True)
    assert len(graph.nodes()) == 4

    pipeline = Pipeline([f, g])
    pipeline.update_scope("foo", inputs={"a", "b"}, outputs={"c", "d"})

    with pytest.raises(
        ValueError,
        match="Cannot combine MapSpecs with different input and output mappings",
    ):
        pipeline.nest_funcs({"foo.c", "foo.d"})
