from __future__ import annotations

from collections import defaultdict

import networkx as nx

import pipefunc
from pipefunc.lazy import TaskGraph, _LazyFunction, construct_dag


def test_construct_dag():
    assert pipefunc.lazy._TASK_GRAPH is None
    with construct_dag() as dag:
        assert isinstance(dag, TaskGraph)
        assert isinstance(dag.graph, nx.DiGraph)
        assert dag.mapping == {}
        assert pipefunc.lazy._TASK_GRAPH is dag
    assert pipefunc.lazy._TASK_GRAPH is None


def test_lazy_function_without_dag():
    def func(x):
        return x + 1

    lazy_func = _LazyFunction(func, args=(1,))
    assert lazy_func.evaluate() == 2


def test_lazy_function_with_dag():
    def func(x):
        return x + 1

    with construct_dag() as dag:
        lazy_func = _LazyFunction(func, args=(1,))

    assert lazy_func.evaluate() == 2
    assert dag.graph.nodes[lazy_func._id]["lazy_func"] == lazy_func
    assert dag.mapping[lazy_func._id] == lazy_func


def test_lazy_function_dependencies():
    def func1(x):
        return x + 1

    def func2(y):
        return y * 2

    with construct_dag() as dag:
        lazy_func1 = _LazyFunction(func1, args=(1,))
        lazy_func2 = _LazyFunction(func2, args=(lazy_func1,))

    assert lazy_func2.evaluate() == 4
    assert dag.graph.has_edge(lazy_func1._id, lazy_func2._id)


def test_lazy_function_list_dependencies():
    def func1(x):
        return x + 1

    def func2(y):
        return y * 2

    def func3(z):
        return sum(z)

    with construct_dag() as dag:
        lazy_func1 = _LazyFunction(func1, args=(1,))
        lazy_func2 = _LazyFunction(func2, args=(2,))
        lazy_func3 = _LazyFunction(func3, args=([lazy_func1, lazy_func2],))

    assert lazy_func3.evaluate() == 6
    assert dag.graph.has_edge(lazy_func1._id, lazy_func3._id)
    assert dag.graph.has_edge(lazy_func2._id, lazy_func3._id)


def test_lazy_pipeline():
    @pipefunc.pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc.pipefunc(output_name="d")
    def f2(b, c, x=1):
        return b * c * x

    @pipefunc.pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    pipeline = pipefunc.Pipeline([f1, f2, f3], lazy=True)
    f = pipeline.func("e")
    with construct_dag() as dag:
        r1 = f(a=1, b=2)
        r3 = pipeline(a=1, b=2)
        r2 = f.call_full_output(a=1, b=2)
    assert dag.mapping
    assert dag.cache.cache
    assert r1.evaluate() == 18
    assert r2["c"].evaluate() == 3
    assert r2["d"].evaluate() == 6
    assert r2["e"].evaluate() == 18
    assert pipefunc.lazy.evaluate_lazy(r2) == {"a": 1, "b": 2, "c": 3, "d": 6, "e": 18}
    assert r3.evaluate() == 18
    assert str(r1) == "f3(c=f1(a=1, b=2), d=f2(b=2, c=f1(a=1, b=2), x=1), x=1)"


def test_running_dag_pipeline():
    @pipefunc.pipefunc(output_name="c")
    def f1(a, b):
        return a + b

    @pipefunc.pipefunc(output_name="d")
    def f2(b, c, x=1):
        return b * c * x

    @pipefunc.pipefunc(output_name="e")
    def f3(c, d, x=1):
        return c * d * x

    pipeline = pipefunc.Pipeline([f1, f2, f3], lazy=True)
    f = pipeline.func("e")
    assert not isinstance(pipeline._current_cache(), pipefunc._cache.SimpleCache)
    with construct_dag() as dag:
        cache = pipeline._current_cache()
        assert isinstance(cache, pipefunc._cache.SimpleCache)
        assert not cache.cache
        f(a=1, b=2)
        f(a=1, b=2)
        assert cache.cache
        assert cache is dag.cache

    assert dag.mapping
    expected = [
        ("c", (("a", 1), ("b", 2))),
        ("d", (("a", 1), ("b", 2), ("x", 1))),
        ("e", (("a", 1), ("b", 2), ("x", 1))),
    ]
    assert list(dag.cache.cache) == expected

    # Test doing something with the graph. Note that is not a good way of using the graph!
    # I have yet to figure out how to best use the graph.
    gens = list(nx.topological_generations(dag.graph))
    assert len(gens) == 3

    results = defaultdict(list)
    for gen in gens:
        for node in gen:
            lazy_func = dag.mapping[node]
            name = lazy_func.func.__name__
            r = lazy_func.evaluate()
            kwargs = pipefunc.lazy.evaluate_lazy(lazy_func.kwargs)
            results[name].append((kwargs, r))

    assert results == {
        "f1": [({"a": 1, "b": 2}, 3)],
        "f2": [({"b": 2, "c": 3, "x": 1}, 6)],
        "f3": [({"c": 3, "d": 6, "x": 1}, 18)],
    }


def test_evaluate_lazy_set():
    def func1(x):
        return x + 1

    def func2(z):
        return sum(z)

    lazy_func1 = _LazyFunction(func1, args=(1,))
    lazy_func2 = _LazyFunction(func2, args=({lazy_func1, lazy_func1},))
    assert lazy_func2.evaluate() == 2
