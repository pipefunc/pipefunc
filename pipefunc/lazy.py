"""Provides the `pipefunc.lazy` module, which contains functions for lazy evaluation."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, NamedTuple

import networkx as nx

from pipefunc._cache import SimpleCache
from pipefunc._utils import format_function_call

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


class _LazyFunction:
    """Lazy function wrapper for deferred evaluation of a function."""

    __slots__ = [
        "func",
        "args",
        "kwargs",
        "_result",
        "_evaluated",
        "_id",
    ]

    _counter = 0

    def __init__(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}

        self._result = None
        self._evaluated = False

        self._id = _LazyFunction._counter
        _LazyFunction._counter += 1

        if _TASK_GRAPH is not None:
            _TASK_GRAPH.graph.add_node(self._id, lazy_func=self)
            _TASK_GRAPH.mapping[self._id] = self

            def add_edge(arg: Any) -> None:
                if isinstance(arg, _LazyFunction):
                    _TASK_GRAPH.graph.add_edge(arg._id, self._id)
                elif isinstance(arg, Iterable):
                    for item in arg:
                        if isinstance(item, _LazyFunction):
                            _TASK_GRAPH.graph.add_edge(item._id, self._id)

            for arg in self.args:
                add_edge(arg)

            if kwargs is not None:
                for arg in kwargs.values():
                    add_edge(arg)

    def evaluate(self) -> Any:
        """Evaluate the lazy function and return the result."""
        if self._evaluated:
            return self._result
        args = evaluate_lazy(self.args)
        kwargs = evaluate_lazy(self.kwargs)
        result = self.func(*args, **kwargs)
        self._result = result
        self._evaluated = True
        return result

    def __repr__(self) -> str:
        from pipefunc._pipefunc import PipeFunc

        func = str(self.func.__name__) if isinstance(self.func, PipeFunc) else str(self.func)
        return format_function_call(func, self.args, self.kwargs)


class TaskGraph(NamedTuple):
    """A named tuple representing a task graph."""

    graph: nx.DiGraph
    mapping: dict[int, _LazyFunction]
    cache: SimpleCache


@contextlib.contextmanager
def construct_dag() -> Generator[TaskGraph, None, None]:
    """Create a directed acyclic graph (DAG) for a pipeline."""
    global _TASK_GRAPH
    _TASK_GRAPH = TaskGraph(nx.DiGraph(), {}, SimpleCache())
    try:
        yield _TASK_GRAPH
    finally:
        _TASK_GRAPH = None


_TASK_GRAPH: TaskGraph | None = None


def task_graph() -> TaskGraph | None:
    """Return the task graph."""
    return _TASK_GRAPH


def evaluate_lazy(x: Any) -> Any:
    """Evaluate a lazy object."""
    if isinstance(x, _LazyFunction):
        return x.evaluate()
    if isinstance(x, dict):
        return {k: evaluate_lazy(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(evaluate_lazy(v) for v in x)
    if isinstance(x, list):
        return [evaluate_lazy(v) for v in x]
    if isinstance(x, set):
        return {evaluate_lazy(v) for v in x}
    return x
