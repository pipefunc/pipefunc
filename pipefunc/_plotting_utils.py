from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx

from pipefunc._pipefunc import NestedPipeFunc, PipeFunc, _validate_combinable_mapspecs
from pipefunc._utils import at_least_tuple

if TYPE_CHECKING:
    from collections.abc import Sequence

ScopeName = str


class CollapsedScope(NestedPipeFunc):
    """A collapsed scope in the pipeline graph."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


def _unique_output_scope(func: PipeFunc) -> ScopeName | None:
    """Determine the scope of the output of a node."""
    assert isinstance(func, PipeFunc)
    scopes = []
    for output_name in at_least_tuple(func.output_name):
        if "." in output_name:
            scopes.append(output_name.split(".")[0])
        else:
            scopes.append(None)
    if len(set(scopes)) == 1 and scopes[0] is not None:
        return scopes[0]
    return None


def _functions_from_graph(graph: nx.DiGraph) -> list[PipeFunc]:
    return [node for node in graph.nodes() if isinstance(node, PipeFunc)]


def _get_collapsed_scope_graph(
    graph: nx.DiGraph,
    scopes_to_collapse: Literal[True] | Sequence[ScopeName],
) -> nx.DiGraph:
    """Collapse scopes in the graph if requested.

    Ensures that no cycles are introduced.
    """
    from pipefunc._pipeline._base import Pipeline

    # 1. Group functions by scope
    grouped_funcs: dict[ScopeName, list[PipeFunc]] = defaultdict(list)
    other_funcs: list[PipeFunc] = []

    for func in _functions_from_graph(graph):
        scope = _unique_output_scope(func)
        if scope is not None and (scopes_to_collapse is True or scope in scopes_to_collapse):
            grouped_funcs[scope].append(func)
        else:
            other_funcs.append(func)

    # Remove groups with just one function
    for scope in list(grouped_funcs.keys()):
        if len(grouped_funcs[scope]) == 1:
            func = grouped_funcs.pop(scope)[0]
            other_funcs.append(func)

    if not grouped_funcs:
        return graph

    # 2. Check each scope group for potential cycles
    new_pipeline_funcs = other_funcs[:]

    for scope, funcs_in_scope in grouped_funcs.items():
        if _would_create_cycle(graph, funcs_in_scope) or not _is_combinable_mapspecs(
            funcs_in_scope,
        ):
            new_pipeline_funcs.extend(funcs_in_scope)
        else:
            collapsed = CollapsedScope(funcs_in_scope, function_name=scope)
            new_pipeline_funcs.append(collapsed)

    collapsed_pipeline = Pipeline(new_pipeline_funcs)  # type: ignore[arg-type]
    return collapsed_pipeline.graph


def _is_combinable_mapspecs(funcs: list[PipeFunc]) -> bool:
    """Check if the MapSpecs of the functions in the list are combinable."""
    mapspecs = [f.mapspec for f in funcs if f.mapspec is not None]
    if len(mapspecs) <= 1:
        return True
    try:
        _validate_combinable_mapspecs(mapspecs)  # type: ignore[arg-type]
    except ValueError:
        return False
    return True


def _would_create_cycle(graph: nx.DiGraph, funcs_to_collapse: list[PipeFunc]) -> bool:
    """Determines if collapsing the specified functions would create a cycle."""
    if len(funcs_to_collapse) <= 1:
        return False

    # Create a temporary "merged node" in a copy of the graph
    temp_graph = graph.copy()
    merged_node = "TEMP_MERGED_NODE"
    temp_graph.add_node(merged_node)

    # Connect all incoming edges to the merged node
    for func in funcs_to_collapse:
        for predecessor in graph.predecessors(func):
            if predecessor not in funcs_to_collapse:
                temp_graph.add_edge(predecessor, merged_node)

    # Connect all outgoing edges from the merged node
    for func in funcs_to_collapse:
        for successor in graph.successors(func):
            if successor not in funcs_to_collapse:
                temp_graph.add_edge(merged_node, successor)

    # Remove the original nodes
    for func in funcs_to_collapse:
        if temp_graph.has_node(func):  # Check needed due to previous removals
            temp_graph.remove_node(func)

    # Check if the resulting graph has cycles
    try:
        nx.find_cycle(temp_graph, orientation="original")
        return True  # Cycle found  # noqa: TRY300
    except nx.NetworkXNoCycle:
        return False  # No cycle


def maybe_collapse_scope(
    graph: nx.DiGraph,
    collapse_scopes: bool | Sequence[str],
) -> nx.DiGraph:
    """Return a new Pipeline instance with collapsed scopes if requested."""
    if collapse_scopes:
        return _get_collapsed_scope_graph(graph, collapse_scopes)
    return graph
