from __future__ import annotations

import inspect
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from pipefunc._utils import at_least_tuple

if TYPE_CHECKING:
    import sys

    if sys.version_info < (3, 9):  # pragma: no cover
        from typing import Callable
    else:
        from collections.abc import Callable

    import networkx as nx

    from pipefunc._pipefunc import PipeFunc


def _wrap_dict_to_tuple(
    func: Callable[..., Any],
    inputs: tuple[str, ...],
    output_name: str | tuple[str, ...],
) -> Callable[..., Any]:
    sig = inspect.signature(func)
    new_params = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in inputs
    ]
    new_sig = sig.replace(parameters=new_params)

    def call(*args: Any, **kwargs: Any) -> Any:
        """Call the pipeline function with the root arguments."""
        bound = new_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        r = func(**bound.arguments)
        if isinstance(output_name, tuple):
            return tuple(r[k] for k in output_name)
        return r[output_name]

    call.__signature__ = new_sig  # type: ignore[attr-defined]

    return call


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
    Dict[PipeFunc, Set[PipeFunc]]
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


def _get_signature(
    combinable_nodes: dict[PipeFunc, set[PipeFunc]],
    graph: nx.DiGraph,
) -> tuple[dict[PipeFunc, set[str]], dict[PipeFunc, set[str]]]:
    """Retrieve the inputs and outputs for the signature of the combinable nodes.

    This function generates a mapping of the inputs and outputs required for
    each node in the combinable_nodes dictionary. For each node, it collects
    the outputs of all nodes it depends on and the parameters it and its
    dependent nodes require. In addition, it considers additional outputs
    based on the dependencies in the graph. It then filters these lists to
    ensure that no parameter is considered an output and no output is
    considered a parameter.

    Parameters
    ----------
    combinable_nodes
        Dictionary containing the nodes that can be combined together.
        The keys of the dictionary are the nodes that can be combined,
        and the values are sets of nodes that they depend on.
    graph
        The directed graph of the pipeline functions. Each node represents a
        function, and each edge represents a dependency relationship between
        functions.

    Returns
    -------
    all_inputs : dict[PipeFunc, set[str]]
        Dictionary where keys are nodes and values are sets of parameter
        names that the node and its dependent nodes require.
    all_outputs : dict[PipeFunc, set[str]]
        Dictionary where keys are nodes and values are sets of output names
        that the node and its dependent nodes produce, plus additional output
        names based on the dependency relationships in the graph.

    """
    all_inputs = {}
    all_outputs = {}
    for node, to_replace in combinable_nodes.items():
        outputs = set(at_least_tuple(node.output_name))
        parameters = set(node.parameters)
        additional_outputs = set()  # parameters that are outputs to other functions
        for f in to_replace:
            outputs |= set(at_least_tuple(f.output_name))
            parameters |= set(f.parameters)
            for successor in graph.successors(f):
                if successor not in to_replace and successor != node:
                    edge = graph.edges[f, successor]
                    additional_outputs |= set(at_least_tuple(edge["arg"]))
        all_outputs[node] = (outputs - parameters) | additional_outputs
        all_inputs[node] = parameters - outputs
    return all_inputs, all_outputs
