from __future__ import annotations

import html
import inspect
import re
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Union

import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout

from pipefunc._pipefunc import NestedPipeFunc, PipeFunc
from pipefunc._pipeline._base import _Bound, _Resources
from pipefunc._plotting_utils import (
    CollapsedScope,
    collapsed_scope_graph,
    find_exclusive_parameters,
)
from pipefunc._utils import at_least_tuple, is_running_in_ipynb, requires
from pipefunc.typing import NoAnnotation, type_as_string

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import graphviz
    import holoviews as hv
    import IPython.display
    import matplotlib.pyplot as plt


_empty = inspect.Parameter.empty
MAX_LABEL_LENGTH = 20


def _get_graph_layout(graph: nx.DiGraph) -> dict:
    """Gets the layout of the graph using Graphviz if available, otherwise defaults to a spring layout."""
    try:
        return graphviz_layout(graph, prog="dot")  # requires pygraphviz
    except ImportError:  # pragma: no cover
        warnings.warn(
            "pygraphviz is not installed. Using spring layout instead.",
            ImportWarning,
            stacklevel=2,
        )
        return nx.spring_layout(graph)


def _trim(s: Any, max_len: int = MAX_LABEL_LENGTH) -> str:
    """Trims a string to a specified maximum length, adding ellipses if trimmed."""
    s = str(s)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


@dataclass
class _Nodes:
    """Contains lists of different node types for the purpose of graph visualization."""

    arg: list[str] = field(default_factory=list)
    func: list[PipeFunc] = field(default_factory=list)
    nested_func: list[NestedPipeFunc] = field(default_factory=list)
    collapsed_scope: list[CollapsedScope] = field(default_factory=list)
    bound: list[_Bound] = field(default_factory=list)
    resources: list[_Resources] = field(default_factory=list)

    def append(self, node: Any) -> None:
        """Appends a node to the appropriate list based on its type."""
        if isinstance(node, str):
            self.arg.append(node)
        elif isinstance(node, CollapsedScope):
            self.collapsed_scope.append(node)
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

    @classmethod
    def from_graph(cls, graph: nx.DiGraph) -> _Nodes:
        """Extracts nodes from a graph."""
        nodes = cls()
        for node in graph.nodes:
            nodes.append(node)
        return nodes


def _all_type_annotations(graph: nx.DiGraph) -> dict[str, type]:
    """Returns a dictionary of all type annotations from the graph nodes."""
    hints: dict[str, type] = {}
    for node in graph.nodes:
        if isinstance(node, PipeFunc):
            outputs = {k: v for k, v in node.output_annotation.items() if v is not NoAnnotation}
            hints |= outputs | node.parameter_annotations
    return hints


class _Labels(NamedTuple):
    outputs: dict
    outputs_mapspec: dict
    inputs: dict
    inputs_mapspec: dict
    bound: dict
    resources: dict
    arg_mapspec: dict

    @classmethod
    def from_graph(cls, graph: nx.DiGraph) -> _Labels:  # noqa: PLR0912
        """Prepares the edge labels for graph visualization."""
        outputs = {}
        inputs = {}
        bound = {}
        outputs_mapspec = {}
        inputs_mapspec = {}
        resources = {}
        arg_mapspec = {}

        for edge, attrs in graph.edges.items():
            a, b = edge
            if isinstance(a, str):
                assert not isinstance(b, str)  # `b` is PipeFunc
                default_value = b.defaults.get(a, _empty)
                if b.mapspec and a in b.mapspec.input_names:
                    spec = next(i for i in b.mapspec.inputs if i.name == a)
                    inputs_mapspec[edge] = str(spec)
                    arg_mapspec[a] = str(spec)
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
                resources[edge] = a.name

        return cls(outputs, outputs_mapspec, inputs, inputs_mapspec, bound, resources, arg_mapspec)


NodeType = Union[str, PipeFunc, _Bound, _Resources, NestedPipeFunc, CollapsedScope, list[str]]


def _generate_node_label(
    node: NodeType,
    hints: dict[str, type],
    defaults: dict[str, Any] | None,
    arg_mapspec: dict[str, str],
    include_full_mapspec: bool,  # noqa: FBT001
    target_func: PipeFunc | None = None,
    is_grouped: bool = False,
) -> str:
    """Generate a Graphviz-compatible HTML-like label for a graph node including type annotations and default values."""

    def _format_type_and_default(
        name: str,
        type_string: str | None,
        default_value: Any = _empty,
    ) -> str:
        """Formats the part of the label with type and default value."""
        parts = [f"<b>{html.escape(name)}</b>"]

        if type_string:
            type_string = html.escape(_trim(type_string))
            parts.append(f" : <i>{type_string}</i>")

        if default_value is not _empty:
            default_value = html.escape(_trim(default_value))
            parts.append(f" = {default_value}")

        return " ".join(parts)

    if is_grouped:  # Handle grouped parameters
        assert isinstance(node, list)
        assert target_func is not None
        title = f"Grouped Inputs ({target_func.__name__})"
        label = f'<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0"><TR><TD BORDER="0"><B>{title}</B></TD></TR><HR/>'
        for param_name in node:
            type_string = type_as_string(hints[param_name]) if param_name in hints else None
            default_value = defaults.get(param_name, _empty) if defaults else _empty
            mapspec = arg_mapspec.get(param_name)
            display_name = mapspec or param_name
            formatted_label = _format_type_and_default(display_name, type_string, default_value)
            # Use ALIGN="LEFT" for better readability in the table
            label += f'<TR><TD ALIGN="LEFT">{formatted_label}</TD></TR>'
        label += "</TABLE>"
        return label

    if isinstance(node, _Bound | _Resources):
        return f"<b>{html.escape(node.name)}</b>"

    if isinstance(node, str):
        type_string = type_as_string(hints[node]) if node in hints else None
        default_value = defaults.get(node, _empty) if defaults else _empty
        mapspec = arg_mapspec.get(node)
        label = _format_type_and_default(mapspec or node, type_string, default_value)

    elif isinstance(node, PipeFunc | NestedPipeFunc | CollapsedScope):
        name = str(node).split(" → ")[0]
        if isinstance(node, CollapsedScope):
            assert node.function_name is not None
            content = f"Scope: {html.escape(node.function_name)}"
        else:
            content = html.escape(name)
        label = f'<TABLE BORDER="0"><TR><TD><B>{content}</B></TD></TR><HR/>'

        for i, output in enumerate(at_least_tuple(node.output_name)):
            name = str(node.mapspec.outputs[i]) if node.mapspec else output
            h = node.output_annotation.get(output)
            type_string = type_as_string(h) if h is not NoAnnotation else None
            default_value = defaults.get(output, _empty) if defaults else _empty
            formatted_label = _format_type_and_default(name, type_string, default_value)
            label += f"<TR><TD>{formatted_label}</TD></TR>"
        if include_full_mapspec and node.mapspec:
            s = html.escape(str(node.mapspec))
            label += f'<HR/><TR><TD><FONT FACE="Courier New">{s}</FONT></TD></TR>'

        label += "</TABLE>"
    else:  # pragma: no cover
        msg = f"Unexpected node type: {type(node)}"
        raise TypeError(msg)

    return label


# Matplotlib colors to hex
_COLORS = {
    "skyblue": "#87CEEB",
    "blue": "#0000FF",
    "lightgreen": "#90EE90",
    "darkgreen": "#006400",
    "red": "#FF0000",
    "orange": "#FFA500",
}


@dataclass
class GraphvizStyle:
    """Dataclass for storing style used in the graphviz visualization."""

    # Nodes
    arg_node_color: str = _COLORS["lightgreen"]
    func_node_color: str = _COLORS["skyblue"]
    nested_func_node_color: str = _COLORS["red"]
    bound_node_color: str = _COLORS["red"]
    resources_node_color: str = _COLORS["orange"]
    collapsed_scope_node_color: str = _COLORS["blue"]
    # Style for grouped params node
    grouped_args_node_color: str = _COLORS["lightgreen"]
    # Edges
    arg_edge_color: str | None = None  # default is arg_node_color
    output_edge_color: str | None = None  # default is func_node_color
    bound_edge_color: str | None = None  # default is bound_node_color
    resources_edge_color: str | None = None  # default is resources_node_color
    input_mapspec_edge_color: str = _COLORS["darkgreen"]
    output_mapspec_edge_color: str = _COLORS["blue"]
    # Style for edge coming from grouped params node
    grouped_args_edge_color: str | None = None  # default uses node color
    # Font
    font_name: str = "Helvetica"
    font_size: int = 12
    edge_font_size: int = 10
    legend_font_size: int = 20
    font_color: str = "black"
    # Background
    legend_background_color: str = "lightgrey"
    background_color: str | None = None
    # Other
    legend_border_color: str = "black"


def visualize_graphviz(  # noqa: PLR0912, C901, PLR0915
    graph: nx.DiGraph,
    defaults: dict[str, Any] | None = None,
    *,
    figsize: tuple[int, int] | int | None = None,
    collapse_scopes: bool | Sequence[str] = False,
    filename: str | Path | None = None,
    style: GraphvizStyle | None = None,
    orient: Literal["TB", "LR", "BT", "RL"] = "LR",
    graphviz_kwargs: dict[str, Any] | None = None,
    show_legend: bool = True,
    include_full_mapspec: bool = False,
    return_type: Literal["graphviz", "html"] | None = None,
) -> graphviz.Digraph | IPython.display.HTML:
    """Visualize the pipeline as a directed graph using Graphviz.

    Parameters
    ----------
    graph
        The directed graph representing the pipeline.
    defaults
        Default values for the function arguments.
    figsize
        The width and height of the figure in inches.
        If a single integer is provided, the figure will be a square.
        If ``None``, the size will be determined automatically.
    collapse_scopes
        Whether to collapse scopes in the graph.
        If ``True``, scopes are collapsed into a single node.
        If a sequence of scope names, only the specified scopes are collapsed.
    filename
        The filename to save the figure to, if provided.
    style
        Style for the graph visualization.
    orient
        Graph orientation: 'TB', 'LR', 'BT', 'RL'.
    graphviz_kwargs
        Graphviz-specific keyword arguments for customizing the graph's appearance.
    show_legend
        Whether to show the legend in the graph visualization.
    include_full_mapspec
        Whether to include the full mapspec as a separate line in the `PipeFunc` labels.
    return_type
        The format to return the visualization in.
        If ``'html'``, the visualization is returned as a `IPython.display.html`,
        if ``'graphviz'``, the `graphviz.Digraph` object is returned.
        If ``None``, the format is ``'html'`` if running in a Jupyter notebook,
        otherwise ``'graphviz'``.

    Returns
    -------
    graphviz.Digraph
        The resulting Graphviz Digraph object.

    """
    requires("graphviz", reason="visualize_graphviz", extras="plotting")
    import graphviz

    if collapse_scopes:
        graph = collapsed_scope_graph(graph, collapse_scopes)

    if style is None:
        style = GraphvizStyle()
    if graphviz_kwargs is None:
        graphviz_kwargs = {}

    # Find exclusive parameters for grouping
    grouped_params, params_to_group = find_exclusive_parameters(graph)
    grouped_node_names = {
        target_func: f"__group_{id(target_func)}" for target_func in grouped_params
    }

    graph_attr: dict[str, Any] = {
        "rankdir": orient,
        "fontsize": str(style.font_size),
        "fontname": style.font_name,
    }
    if figsize:
        graph_attr["size"] = (
            f"{figsize[0]},{figsize[1]}" if isinstance(figsize, tuple) else f" {figsize},{figsize}"
        )
        graph_attr["ratio"] = "fill"
    if style.background_color:
        graph_attr["bgcolor"] = style.background_color

    # Graphviz Setup
    digraph = graphviz.Digraph(
        comment="Graph Visualization",
        graph_attr=graph_attr,
        node_attr={
            "shape": "plaintext",
            "fontname": style.font_name,
            "fontsize": str(style.font_size),
            "style": "filled",
        },
        edge_attr={
            "fontname": style.font_name,
            "fontsize": str(style.edge_font_size),
        },
        **graphviz_kwargs,
    )

    hints = _all_type_annotations(graph)
    edge_labels_info = _Labels.from_graph(graph)

    # Node defaults and configurations
    node_defaults = {
        "width": "0",
        "height": "0",
        "margin": "0.1",
        "penwidth": "1",
        "color": "black",
    }

    # Node configurations for styling and legend
    node_configs = {
        "Argument": {
            "fillcolor": style.arg_node_color,
            "shape": "rectangle",
            "style": "filled,dashed",
        },
        "Grouped Inputs": {
            "fillcolor": style.grouped_args_node_color,
            "shape": "note",
            "style": "filled",
        },
        "PipeFunc": {
            "fillcolor": style.func_node_color,
            "shape": "box",
            "style": "filled,rounded",
        },
        "NestedPipeFunc": {
            "fillcolor": style.func_node_color,
            "shape": "box",
            "style": "filled,rounded",
            "color": style.nested_func_node_color,
        },
        "CollapsedScope": {
            "fillcolor": style.func_node_color,
            "shape": "box",
            "style": "filled,rounded",
            "color": style.collapsed_scope_node_color,
            "penwidth": "2.0",
            "peripheries": "2",
        },
        "Bound": {
            "fillcolor": style.bound_node_color,
            "shape": "hexagon",
            "style": "filled",
        },
        "Resources": {
            "fillcolor": style.resources_node_color,
            "shape": "hexagon",
            "style": "filled",
        },
    }
    legend_items = {}  # For legend

    # Create grouped parameter nodes
    if grouped_params:
        legend_items["Grouped Inputs"] = node_configs["Grouped Inputs"]

    for target_func, params in grouped_params.items():
        grouped_node_name = grouped_node_names[target_func]
        label = _generate_node_label(
            params,
            hints,
            defaults,
            edge_labels_info.arg_mapspec,
            False,  # include_full_mapspec not relevant for grouped params
            target_func,
            True,  # is_grouped
        )
        attribs = {**node_defaults, **node_configs["Grouped Inputs"], "label": label}
        digraph.node(grouped_node_name, **attribs)

    # Create regular nodes (skip grouped parameters)
    for node in graph.nodes:
        # Skip parameters that are part of a group
        if isinstance(node, str) and node in params_to_group:
            continue

        # Determine node type and configuration
        if isinstance(node, str):
            node_type_key = "Argument"
        elif isinstance(node, CollapsedScope):
            node_type_key = "CollapsedScope"
        elif isinstance(node, NestedPipeFunc):
            node_type_key = "NestedPipeFunc"
        elif isinstance(node, PipeFunc):
            node_type_key = "PipeFunc"
        elif isinstance(node, _Bound):
            node_type_key = "Bound"
        elif isinstance(node, _Resources):
            node_type_key = "Resources"
        else:
            continue  # Should not happen

        # Add to legend items if not already present
        if node_type_key not in legend_items:
            legend_items[node_type_key] = node_configs[node_type_key]

        # Generate node label
        label = _generate_node_label(
            node,
            hints,
            defaults,
            edge_labels_info.arg_mapspec,
            include_full_mapspec,
            None,  # target_func
            False,  # is_grouped
        )

        # Handle HTML-like labels properly
        if not label.startswith("<"):
            label = f"<{label}>"

        # Create the node
        attribs = {**node_defaults, **node_configs[node_type_key], "label": label}
        digraph.node(str(node), **attribs)

    # Add edges with appropriate colors and labels
    edge_colors = {
        "output": style.output_edge_color or style.func_node_color,
        "output_mapspec": style.output_mapspec_edge_color,
        "input": style.arg_edge_color or style.arg_node_color,
        "input_mapspec": style.input_mapspec_edge_color,
        "bound": style.bound_edge_color or style.bound_node_color,
        "resources": style.resources_edge_color or style.resources_node_color,
        "grouped": style.grouped_args_edge_color or style.grouped_args_node_color,
    }

    # Track processed group edges to avoid duplicates
    processed_grouped_edges = set()

    for a, b in graph.edges:
        # Skip individual parameters that are part of a group
        if isinstance(a, str) and a in params_to_group:
            # Get the target function
            target_func = b
            assert isinstance(target_func, PipeFunc)

            # Get the group node name
            group_node_name = grouped_node_names[target_func]

            # Check if we've already processed this group->target edge
            edge_key = (group_node_name, str(target_func))
            if edge_key in processed_grouped_edges:
                continue

            # Add an edge from the group node to the target function
            edge_color = edge_colors["grouped"]

            # Create a label that shows all params in the group
            label = ", ".join(grouped_params[target_func])
            if len(label) > 30:
                label = f"{len(grouped_params[target_func])} inputs"

            digraph.edge(
                group_node_name,
                str(target_func),
                color=edge_color,
                label=label,
                tooltip=f"<<b>{html.escape(label)}</b>>",
                penwidth="1.01",
                fontcolor="black" if len(label) < 20 else "transparent",
            )

            processed_grouped_edges.add(edge_key)
            continue

        # Normal edge processing for non-grouped parameters
        if isinstance(a, PipeFunc):
            edge_color = edge_colors["output"]
            if (a, b) in edge_labels_info.outputs_mapspec:
                edge_color = edge_colors["output_mapspec"]
                label = edge_labels_info.outputs_mapspec[(a, b)]
            elif (a, b) in edge_labels_info.outputs:
                label = edge_labels_info.outputs[(a, b)]
            else:
                label = ", ".join(at_least_tuple(graph.edges[(a, b)].get("arg", "?")))
        elif isinstance(a, str):
            edge_color = edge_colors["input"]
            if (a, b) in edge_labels_info.inputs_mapspec:
                edge_color = edge_colors["input_mapspec"]
                label = edge_labels_info.inputs_mapspec[(a, b)]
            elif (a, b) in edge_labels_info.inputs:
                label = edge_labels_info.inputs[(a, b)]
            else:
                label = a
        elif isinstance(a, _Bound):
            edge_color = edge_colors["bound"]
            label = edge_labels_info.bound[(a, b)]
        elif isinstance(a, _Resources):
            edge_color = edge_colors["resources"]
            label = edge_labels_info.resources[(a, b)]
        else:
            continue

        digraph.edge(
            str(a),
            str(b),
            color=edge_color,
            label=label,
            tooltip=f"<<b>{html.escape(label)}</b>>",
            penwidth="1.01",
            fontcolor="transparent",
        )

    # Create legend
    if show_legend and legend_items:
        legend_subgraph = graphviz.Digraph(
            name="cluster_legend",
            graph_attr={
                "label": "Legend",
                "rankdir": orient,
                "fontsize": str(style.legend_font_size),
                "fontcolor": style.font_color,
                "fontname": style.font_name,
                "color": style.legend_border_color,
                "style": "filled",
                "fillcolor": style.legend_background_color,
            },
        )

        # Sort legend items for consistent display
        sorted_legend_items = dict(sorted(legend_items.items()))
        prev_node_name = None

        for i, (name, config) in enumerate(sorted_legend_items.items()):
            node_name = f"legend_{i}"

            # Create a copy of the config to modify
            attribs = {**node_defaults, "label": name, **config}

            # Remove margin for legend nodes for more compact display
            if "margin" in attribs:
                del attribs["margin"]

            legend_subgraph.node(node_name, **attribs)

            if prev_node_name is not None:
                legend_subgraph.edge(prev_node_name, node_name, style="invis")

            prev_node_name = node_name

        digraph.subgraph(legend_subgraph)

    if filename is not None:
        name, extension = str(filename).rsplit(".", 1)
        digraph.render(name, format=extension, cleanup=True)

    if return_type is None and is_running_in_ipynb():  # pragma: no cover
        return_type = "html"

    if return_type == "html":
        from IPython.display import HTML

        svg_content = digraph._repr_image_svg_xml()
        html_content = (
            f'<div id="svg-container" style="max-width: 100%;">{svg_content}</div>'
            "<style>#svg-container svg {max-width: 100%; height: auto;}</style>"
        )

        return HTML(html_content)

    return digraph


def visualize_matplotlib(
    graph: nx.DiGraph,
    figsize: tuple[int, int] | int = (10, 10),
    filename: str | Path | None = None,
    func_node_colors: str | list[str] | None = None,
) -> plt.Figure:
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
        The color of the function nodes.

    Returns
    -------
        The resulting Matplotlib figure.

    """
    requires("matplotlib", reason="visualize_matplotlib", extras="plotting")
    import matplotlib.pyplot as plt

    pos = _get_graph_layout(graph)
    if isinstance(figsize, int):
        figsize = (figsize, figsize)
    fig = plt.figure(figsize=figsize)

    nodes = _Nodes.from_graph(graph)
    colors_shapes_edgecolors = [
        (nodes.arg, "lightgreen", "s", None),
        (nodes.func, func_node_colors or "skyblue", "o", None),
        (nodes.nested_func, func_node_colors or "skyblue", "o", "red"),
        (nodes.bound, "red", "h", None),
        (nodes.resources, "C1", "p", None),
    ]
    for _nodes, color, shape, edgecolor in colors_shapes_edgecolors:
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
        """Add mapspec to function output if applicable."""
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
        nx.draw_networkx_labels(graph, pos, labels, font_size=12)

    nx.draw_networkx_edges(graph, pos, arrows=True, node_size=4000)

    # Add edge labels with function outputs
    labels = _Labels.from_graph(graph)

    for _labels, color in [
        (labels.outputs, "skyblue"),
        (labels.outputs_mapspec, "blue"),
        (labels.inputs, "lightgreen"),
        (labels.inputs_mapspec, "green"),
        (labels.bound, "red"),
    ]:
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=_labels,
            font_size=12,
            font_color=color,
        )

    plt.axis("off")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return fig


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
    requires("holoviews", "bokeh", reason="visualize_holoviews", extras="plotting")
    import bokeh.plotting
    import holoviews as hv

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
    edges = np.array([(node_index_dict[edge[0]], node_index_dict[edge[1]]) for edge in graph.edges])

    # Create Nodes and Graph
    nodes = hv.Nodes((x, y, node_indices, node_labels, node_types), vdims=["label", "type"])
    graph = hv.Graph((edges, nodes))

    plot_opts = {
        "width": 800,
        "height": 800,
        "padding": 0.1,
        "xaxis": None,
        "yaxis": None,
        "node_color": hv.dim("type").categorize({"str": "lightgreen", "func": "skyblue"}, "gray"),
        "edge_color": "black",
    }

    graph = graph.opts(**plot_opts)

    # Create Labels and add them to the graph
    labels = hv.Labels(graph.nodes, ["x", "y"], "label")
    plot = graph * labels.opts(text_font_size="8pt", text_color="black", bgcolor="white")
    if show:  # pragma: no cover
        bokeh.plotting.show(hv.render(plot))
        return None
    return plot
