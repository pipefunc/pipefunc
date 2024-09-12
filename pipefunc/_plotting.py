from __future__ import annotations

import inspect
import re
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from pipefunc._pipefunc import NestedPipeFunc, PipeFunc
from pipefunc._pipeline import _Bound, _Resources
from pipefunc._utils import at_least_tuple

if TYPE_CHECKING:
    from pathlib import Path

    import graphviz
    import holoviews as hv


_empty = inspect.Parameter.empty
MAX_LABEL_LENGTH = 20


def _get_graph_layout(graph: nx.DiGraph) -> dict:
    try:
        return graphviz_layout(graph, prog="dot")
    except ImportError:  # pragma: no cover
        warnings.warn(
            "Graphviz is not installed. Using spring layout instead.",
            ImportWarning,
            stacklevel=2,
        )
        return nx.spring_layout(graph)


def _trim(s: Any, max_len: int = MAX_LABEL_LENGTH) -> str:
    s = str(s)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


@dataclass
class _Nodes:
    arg: list[str] = field(default_factory=list)
    func: list[PipeFunc] = field(default_factory=list)
    nested_func: list[NestedPipeFunc] = field(default_factory=list)
    bound: list[_Bound] = field(default_factory=list)
    resources: list[_Resources] = field(default_factory=list)

    def append(self, node: Any) -> None:
        if isinstance(node, str):
            self.arg.append(node)
        elif isinstance(node, NestedPipeFunc):
            self.nested_func.append(node)
        elif isinstance(node, PipeFunc):
            self.func.append(node)
        elif isinstance(node, _Bound):
            self.bound.append(node)
        elif isinstance(node, _Resources):
            self.resources.append(node)
        else:  # pragma: no cover
            msg = "Should not happen. Please report this as a bug."
            raise TypeError(msg)


def visualize(  # noqa: C901, PLR0912, PLR0915
    graph: nx.DiGraph,
    figsize: tuple[int, int] | int = (10, 10),
    filename: str | Path | None = None,
    func_node_colors: str | list[str] | None = None,
) -> None:
    """Visualize the pipeline as a directed graph.

    Parameters
    ----------
    graph
        The directed graph representing the pipeline.
    figsize
        The width and height of the figure in inches.
        If a single integer is provided, the figure will be a square.
    filename
        The filename to save the figure to.
    func_node_colors
        The color of the nodes.

    """
    import matplotlib.pyplot as plt

    pos = _get_graph_layout(graph)
    nodes = _Nodes()
    for node in graph.nodes:
        nodes.append(node)
    if isinstance(figsize, int):
        figsize = (figsize, figsize)
    plt.figure(figsize=figsize)

    for _nodes, color, shape, edgecolor in [
        (nodes.arg, "lightgreen", "s", None),
        (nodes.func, func_node_colors or "skyblue", "o", None),
        (nodes.nested_func, func_node_colors or "skyblue", "o", "red"),
        (nodes.bound, "red", "h", None),
        (nodes.resources, "C1", "p", None),
    ]:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=_nodes,
            node_size=4000,
            node_color=color,
            node_shape=shape,
            edgecolors=edgecolor,
        )

    def func_with_mapspec(func: PipeFunc) -> str:
        s = str(func)
        if not func.mapspec:
            return s
        for spec in func.mapspec.outputs:
            # changes e.g., "func(...) -> y" to "func(...) -> y[i]"
            s = re.sub(rf"\b{spec.name}\b", str(spec), s)
        return s

    for labels in [
        {node: node for node in nodes.arg},
        {node: func_with_mapspec(node) for node in nodes.func + nodes.nested_func},
        {node: node.name for node in nodes.bound + nodes.resources},
    ]:
        nx.draw_networkx_labels(
            graph,
            pos,
            labels,
            font_size=12,
        )

    nx.draw_networkx_edges(graph, pos, arrows=True, node_size=4000)

    # Add edge labels with function outputs
    outputs = {}
    inputs = {}
    bound = {}
    outputs_mapspec = {}
    inputs_mapspec = {}

    for edge, attrs in graph.edges.items():
        a, b = edge
        if isinstance(a, str):
            assert not isinstance(b, str)  # `b` is PipeFunc
            default_value = b.defaults.get(a, _empty)
            if b.mapspec and a in b.mapspec.input_names:
                spec = next(i for i in b.mapspec.inputs if i.name == a)
                inputs_mapspec[edge] = str(spec)
            elif default_value is not _empty:
                inputs[edge] = f"{a}={_trim(default_value)}"
            else:
                inputs[edge] = a
        elif isinstance(a, PipeFunc):
            output_str = []
            with_mapspec = False
            for name in at_least_tuple(attrs["arg"]):
                if b.mapspec and name in b.mapspec.input_names:
                    with_mapspec = True
                    spec = next(i for i in b.mapspec.inputs if i.name == name)
                    output_str.append(str(spec))
                else:
                    output_str.append(name)
            if with_mapspec:
                outputs_mapspec[edge] = ", ".join(output_str)
            else:
                outputs[edge] = ", ".join(output_str)
        elif isinstance(a, _Bound):
            bound_value = b._bound[a.name]
            bound[edge] = f"{a.name}={bound_value}"
        else:
            assert isinstance(a, _Resources)

    for labels, color in [
        (outputs, "skyblue"),
        (outputs_mapspec, "blue"),
        (inputs, "lightgreen"),
        (inputs_mapspec, "green"),
        (bound, "red"),
    ]:
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=labels,
            font_size=12,
            font_color=color,
        )

    plt.axis("off")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def visualize_graphviz(
    graph: nx.DiGraph,
    filename: str | Path | None = None,
    func_node_colors: str | list[str] | None = None,
    orient: str = "LR",
    graphviz_kwargs: dict = None,
) -> graphviz.Digraph:
    """Visualize the pipeline as a directed graph using Graphviz.

    Parameters
    ----------
    graph : nx.DiGraph
        The directed graph representing the pipeline.
    filename : str | Path | None
        The filename to save the figure to, if provided.
    func_node_colors : str | list[str] | None
        The colors for function nodes.
    orient : str
        Graph orientation: 'TB', 'LR', 'BT', 'RL'.
    graphviz_kwargs : dict
        Graphviz-specific keyword arguments for customizing the graph's appearance.

    Returns
    -------
    digraph : graphviz.Digraph
        The resulting Graphviz Digraph object.

    """
    import graphviz

    if graphviz_kwargs is None:
        graphviz_kwargs = {}

    # Prepare nodes data
    nodes = _Nodes()
    for node in graph.nodes:
        nodes.append(node)

    # Graphviz Setup
    digraph = graphviz.Digraph(
        comment="Graph Visualization",
        graph_attr={"rankdir": orient},
        node_attr={"shape": "rectangle"},
        **graphviz_kwargs,
    )

    # Add nodes to visual graph
    for nodelist, color, shape, edgecolor, style in [
        (nodes.arg, "lightgreen", "rectangle", None, "filled,dashed"),
        (nodes.func, func_node_colors or "skyblue", "box", None, "rounded,filled"),
        (nodes.nested_func, func_node_colors or "skyblue", "box", "red", "filled"),
        (nodes.bound, "red", "hexagon", None, "filled"),
        (nodes.resources, "orange", "polygon", None, "filled"),
    ]:
        for node in nodelist:
            attribs = {
                "color": color,
                "style": style,
                "shape": shape,
                "width": "0.75",
                "height": "0.5",
            }
            if edgecolor:
                attribs["color"] = edgecolor
            digraph.node(str(node), **attribs)

    # Add edges and labels with function outputs
    outputs = {}
    inputs = {}
    bound = {}
    outputs_mapspec = {}
    inputs_mapspec = {}

    for edge, attrs in graph.edges.items():
        a, b = edge
        if isinstance(a, str):
            assert not isinstance(b, str)  # `b` is PipeFunc
            default_value = b.defaults.get(a, _empty)
            if b.mapspec and a in b.mapspec.input_names:
                spec = next(i for i in b.mapspec.inputs if i.name == a)
                inputs_mapspec[edge] = str(spec)
            elif default_value is not _empty:
                inputs[edge] = f"{a}={_trim(default_value)}"
            else:
                inputs[edge] = a
        elif isinstance(a, PipeFunc):
            output_str = []
            with_mapspec = False
            for name in at_least_tuple(attrs["arg"]):
                if b.mapspec and name in b.mapspec.input_names:
                    with_mapspec = True
                    spec = next(i for i in b.mapspec.inputs if i.name == name)
                    output_str.append(str(spec))
                else:
                    output_str.append(name)
            if with_mapspec:
                outputs_mapspec[edge] = ", ".join(output_str)
            else:
                outputs[edge] = ", ".join(output_str)
        elif isinstance(a, _Bound):
            bound_value = b._bound[a.name]
            bound[edge] = f"{a.name}={bound_value}"
        else:
            assert isinstance(a, _Resources)

    for labels, color in [
        (outputs, "blue"),
        (outputs_mapspec, "darkblue"),
        (inputs, "green"),
        (inputs_mapspec, "darkgreen"),
        (bound, "red"),
    ]:
        for edge, label in labels.items():
            digraph.edge(str(edge[0]), str(edge[1]), label=label, color=color)

    if filename is not None:
        digraph.render(filename, format="png", cleanup=True)

    return digraph


def visualize_holoviews(graph: nx.DiGraph, *, show: bool = False) -> hv.Graph | None:
    """Visualize the pipeline as a directed graph using HoloViews.

    Parameters
    ----------
    graph
        The directed graph representing the pipeline.
    show
        Whether to show the plot. Uses `bokeh.plotting.show(holoviews.render(plot))`.
        If ``False`` the `holoviews.Graph` object is returned.

    """
    import bokeh.plotting
    import holoviews as hv
    import numpy as np

    hv.extension("bokeh")
    pos = _get_graph_layout(graph)

    # Extract node positions and create a list of nodes
    x, y = np.array([pos[node] for node in graph.nodes]).T
    node_indices = range(len(graph.nodes))
    node_types = ["str" if isinstance(node, str) else "func" for node in graph.nodes]
    node_labels = [str(node) for node in graph.nodes]

    # Create a dictionary for quick lookup of indices
    node_index_dict = {node: index for index, node in enumerate(graph.nodes)}

    # Extract edge info using the lookup dictionary
    edges = np.array(
        [(node_index_dict[edge[0]], node_index_dict[edge[1]]) for edge in graph.edges],
    )

    # Create Nodes and Graph
    nodes = hv.Nodes(
        (x, y, node_indices, node_labels, node_types),
        vdims=["label", "type"],
    )
    graph = hv.Graph((edges, nodes))

    plot_opts = {
        "width": 800,
        "height": 800,
        "padding": 0.1,
        "xaxis": None,
        "yaxis": None,
        "node_color": hv.dim("type").categorize(
            {"str": "lightgreen", "func": "skyblue"},
            "gray",
        ),
        "edge_color": "black",
    }

    graph = graph.opts(**plot_opts)

    # Create Labels and add them to the graph
    labels = hv.Labels(graph.nodes, ["x", "y"], "label")
    plot = graph * labels.opts(
        text_font_size="8pt",
        text_color="black",
        bgcolor="white",
    )
    if show:  # pragma: no cover
        bokeh.plotting.show(hv.render(plot))
        return None
    return plot
