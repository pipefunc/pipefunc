import networkx as nx

import pipefunc._lazy
from pipefunc._lazy import TaskGraph, _LazyFunction, construct_dag


def test_construct_dag():
    assert pipefunc._lazy._TASK_GRAPH is None
    with construct_dag() as dag:
        assert isinstance(dag, TaskGraph)
        assert isinstance(dag.graph, nx.DiGraph)
        assert dag.mapping == {}
        assert pipefunc._lazy._TASK_GRAPH is dag
    assert pipefunc._lazy._TASK_GRAPH is None


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
