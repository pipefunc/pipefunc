from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

if TYPE_CHECKING:
    from pathlib import Path

    import holoviews as hv


def _get_graph_layout(graph: nx.DiGraph) -> dict:
    try:
        return graphviz_layout(graph, prog="dot")
    except ImportError:
        warnings.warn(
            "Graphviz is not installed. Using spring layout instead.",
            ImportWarning,
            stacklevel=2,
        )
        return nx.spring_layout(graph)


def visualize(
    graph: nx.DiGraph,
    figsize: tuple[int, int] = (10, 10),
    filename: str | Path | None = None,
) -> None:
    """Visualize the pipeline as a directed graph.

    Parameters
    ----------
    graph
        The directed graph representing the pipeline.
    figsize
        The width and height of the figure in inches, by default (10, 10).
    filename
        The filename to save the figure to, by default None.
    """
    import matplotlib.pyplot as plt

    pos = _get_graph_layout(graph)

    arg_nodes = []
    func_nodes = []
    for node in graph.nodes:
        if isinstance(node, str):
            arg_nodes.append(node)
        else:  # is PipelineFunction
            func_nodes.append(node)

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=arg_nodes,
        node_size=2000,
        node_color="lightgreen",
        node_shape="s",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=func_nodes,
        node_size=4000,
        node_color="skyblue",
        node_shape="o",
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        {node: node for node in arg_nodes},
        font_size=12,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        {node: f"{node!s}" for node in func_nodes},
        font_size=12,
    )

    nx.draw_networkx_edges(graph, pos, arrows=True)

    # Add edge labels with function outputs
    outputs = {}
    inputs = {}
    for edge, attrs in graph.edges.items():
        a, b = edge
        if isinstance(a, str):
            default_value = graph.nodes[a]["default_value"]
            if default_value is not inspect.Parameter.empty:
                inputs[edge] = f"{a}={default_value}"
            else:
                inputs[edge] = a
        else:  # is PipelineFunction
            arg = attrs["arg"]
            if isinstance(arg, tuple):
                arg = ", ".join(arg)
            outputs[edge] = arg

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=outputs,
        font_size=12,
        font_color="skyblue",
    )

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=inputs,
        font_size=12,
        font_color="lightgreen",
    )

    plt.axis("off")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def visualize_holoviews(graph: nx.DiGraph) -> hv.Graph:
    """Visualize the pipeline as a directed graph using HoloViews."""
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
    return graph * labels.opts(
        text_font_size="8pt",
        text_color="black",
        bgcolor="white",
    )
