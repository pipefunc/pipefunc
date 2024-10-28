from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import networkx as nx

from pipefunc._pipefunc import NestedPipeFunc, PipeFunc
from pipefunc._utils import at_least_tuple

if TYPE_CHECKING:
    from collections.abc import Iterable

    import networkx as nx

    from ._base import Pipeline
    from ._types import OUTPUT_TYPE


def simplified_pipeline(
    functions: list[PipeFunc],
    graph: nx.DiGraph,
    all_root_args: dict[OUTPUT_TYPE, tuple[str, ...]],
    node_mapping: dict[OUTPUT_TYPE, PipeFunc | str],
    output_name: OUTPUT_TYPE,
    *,
    conservatively_combine: bool = False,
) -> Pipeline:
    """Simplify pipeline with combined function nodes.

    Generate a simplified version of the pipeline where combinable function
    nodes have been merged into single function nodes.

    This method identifies combinable nodes in the pipeline's execution
    graph (i.e., functions that share the same root arguments) and merges
    them into single `NestedPipeFunc` nodes. This results in a simplified pipeline
    where each key function only depends on nodes that cannot be further
    combined.

    Returns
    -------
        The simplified version of the pipeline.

    """
    from pipefunc import PipeFunc, Pipeline

    func = node_mapping[output_name]
    assert isinstance(func, PipeFunc)
    combinable_nodes = _identify_combinable_nodes(
        func,
        graph,
        all_root_args,
        conservatively_combine=conservatively_combine,
    )
    if not combinable_nodes:
        msg = "No combinable nodes found, the pipeline cannot be simplified."
        raise ValueError(msg)

    # Simplify the combinable_nodes dictionary by replacing any nodes that
    # can be combined with their own dependencies, so that each key in the
    # dictionary only depends on nodes that cannot be further combined.
    combinable_nodes = _combine_nodes(combinable_nodes)
    # Sort to ensure deterministic output
    sorted_nodes = {k: _sort(combinable_nodes[k]) for k in _sort(combinable_nodes.keys())}
    to_combine_flat = _flatten_dict(sorted_nodes)
    funcs = [f for f in functions if f not in to_combine_flat]
    all_inputs = {p for f in funcs for p in f.parameters}
    nested_funcs = [[base, *combine] for base, combine in sorted_nodes.items()]
    for i, to_combine in enumerate(nested_funcs):
        output_name = _output_name(i, nested_funcs, all_inputs)
        nested = NestedPipeFunc(to_combine, output_name=output_name)
        funcs.append(nested)
    return Pipeline(funcs)  # type: ignore[arg-type]


def _output_name(i: int, nested_funcs: list[list[PipeFunc]], all_inputs: set[str]) -> OUTPUT_TYPE:
    to_combine = nested_funcs[i]
    # Get all outputs for functions to combine
    current_outputs = {n for f in to_combine for n in at_least_tuple(f.output_name)}
    # Get inputs to all functions except the current one
    other_inputs = {
        p for j, fs in enumerate(nested_funcs) if j != i for f in fs for p in f.parameters
    } | all_inputs
    outputs_for_others = current_outputs & other_inputs
    base = to_combine[0]
    output_name_set = {*at_least_tuple(base.output_name), *outputs_for_others}
    output_names = tuple(sorted(output_name_set))
    return output_names[0] if len(output_names) == 1 else output_names


def _sort(funcs: Iterable[PipeFunc]) -> list[PipeFunc]:
    return sorted(funcs, key=lambda f: f.output_name)


def _flatten_dict(d: dict[PipeFunc, list[PipeFunc]]) -> list[PipeFunc]:
    return [v for k, lst in d.items() for v in [k, *lst]]


def _identify_combinable_nodes(
    func: PipeFunc,
    graph: nx.DiGraph,
    all_root_args: dict[OUTPUT_TYPE, tuple[str, ...]],
    *,
    conservatively_combine: bool = False,
) -> dict[PipeFunc, set[PipeFunc]]:
    """Identify which function nodes can be combined into a single function.

    This method identifies the PipeFuncs in the execution graph that
    can be combined into a single function. The criterion for combinability
    is that the functions share the same root arguments.

    Parameters
    ----------
    graph
        The directed graph of the pipeline functions and input arguments.
    all_root_args
        A dictionary where each key is the output name of a function in the
        pipeline, and the value is a tuple of root arguments for that function.
    func
        The function in the pipeline function we are starting
        the search from.
    conservatively_combine
        If True, only combine a function node with its predecessors if all
        of its predecessors have the same root arguments as the function
        node itself. If False, combine a function node with its predecessors
        if any of its predecessors have the same root arguments as the
        function node.

    Returns
    -------
        A dictionary where each key is a PipeFunc that can be
        combined with others. The value associated with each key is a set of
        PipeFuncs that can be combined with the key function.

    Notes
    -----
    This function works by performing a depth-first search through the
    pipeline's execution graph. Starting from the PipeFunc
    corresponding to the `output_name`, it goes through each predecessor in
    the graph (functions that need to be executed before the current one).
    For each predecessor function, it recursively checks if it can be
    combined with others by comparing their root arguments.

    If a function's root arguments are identical to the head function's root
    arguments, it is considered combinable and added to the set of
    combinable functions for the head. If `conservatively_combine=True` and
    all predecessor functions are combinable, the head function and its set
    of combinable functions are added to the `combinable_nodes` dictionary.
    If `conservatively_combine=False` and any predecessor function is
    combinable, the head function and its set of combinable functions are
    added to the `combinable_nodes` dictionary.

    The function 'head' in the nested function `_recurse` represents the
    current function being checked in the execution graph.

    """
    # Nested function _recurse performs the depth-first search and updates the
    # `combinable_nodes` dictionary.

    def _recurse(head: PipeFunc) -> None:
        head_args = all_root_args[head.output_name]
        funcs = set()
        i = 0
        for node in graph.predecessors(head):
            if not isinstance(node, PipeFunc):
                continue
            if node.mapspec is not None:
                msg = "`PipeFunc`s with `mapspec` cannot be simplified currently."
                raise NotImplementedError(msg)
            i += 1
            _recurse(node)
            node_args = all_root_args[node.output_name]
            if node_args == head_args:
                funcs.add(node)
        if funcs and (not conservatively_combine or i == len(funcs)):
            combinable_nodes[head] = funcs

    combinable_nodes: dict[PipeFunc, set[PipeFunc]] = {}
    _recurse(func)
    return combinable_nodes


def _combine_nodes(
    combinable_nodes: dict[PipeFunc, set[PipeFunc]],
) -> dict[PipeFunc, set[PipeFunc]]:
    """Reduce the dictionary of combinable nodes to a minimal set.

    The input dictionary `combinable_nodes` indicates which nodes
    (functions in the pipeline) can be combined together. The dictionary
    keys are PipeFunc objects, and the values are sets of
    PipeFunc objects that the key depends on and can be
    combined with. For example,
    if `combinable_nodes = {f6: {f5}, f5: {f1, f4}}`, it means that `f6`
    can be combined with `f5`, and `f5` can be combined with `f1` and `f4`.

    This method simplifies the input dictionary by iteratively checking each
    node in the dictionary to see if it is a dependency of any other nodes.
    If it is, the method replaces that dependency with the node's own
    dependencies and removes the node from the dictionary. For example, if
    `f5` is found to be a dependency of `f6`, then `f5` is replaced by its
    own dependencies `{f1, f4}` in the `f6` entry,  and the `f5` entry is
    removed from the dictionary. This results in a new
    dictionary, `{f6: {f1, f4}}`.

    The aim is to get a dictionary where each node only depends on nodes
    that cannot be further combined. This simplified dictionary is useful for
    constructing a simplified graph of the computation.

    Parameters
    ----------
    combinable_nodes
        A dictionary where the keys are PipeFunc objects, and the
        values are sets of PipeFunc objects that can be combined
        with the key.

    Returns
    -------
        A simplified dictionary where each node only depends on nodes
        that cannot be further combined.

    """
    combinable_nodes = OrderedDict(combinable_nodes)
    for _ in range(len(combinable_nodes)):
        node, deps = combinable_nodes.popitem(last=False)
        added_nodes = []
        for _node, _deps in list(combinable_nodes.items()):
            if node in _deps:
                combinable_nodes[_node] |= deps
                added_nodes.append(_node)
        if not added_nodes:
            combinable_nodes[node] = deps
    return dict(combinable_nodes)


def _func_node_colors(
    functions: list[PipeFunc],
    combinable_nodes: dict[PipeFunc, set[PipeFunc]],
) -> list[str]:
    combinable_nodes = _combine_nodes(combinable_nodes)
    func_node_colors = []
    node_sets = [{k, *v} for k, v in combinable_nodes.items()]
    color_index = len(node_sets)  # for non-combinable nodes
    for node in functions:
        i = next((i for i, nodes in enumerate(node_sets) if node in nodes), None)
        if i is not None:
            func_node_colors.append(f"C{i}")
        else:
            func_node_colors.append(f"C{color_index}")
            color_index += 1
    return func_node_colors
