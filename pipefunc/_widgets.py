from pathlib import Path
from typing import Any

import anywidget
import ipywidgets as widgets
import traitlets


class PipeFuncGraphWidget(anywidget.AnyWidget):
    """A widget for rendering a graphviz graph using d3-graphviz.

    Example:
    -------
    >>> dot_string = "digraph { a -> b; b -> c; c -> a; }"
    >>> pipe_func_graph_widget = PipeFuncGraphWidget(dot_source=dot_string)
    >>> pipe_func_graph_widget

    """

    _esm = Path(__file__).parent / "static" / "graphviz_widget.js"

    _css = """
    #graph {
        margin: auto;
    }
    """

    dot_source = traitlets.Unicode("").tag(sync=True)
    selected_direction = traitlets.Unicode("bidirectional").tag(sync=True)


def graph_widget(dot_string: str = "digraph { a -> b; b -> c; c -> a; }") -> widgets.VBox:
    pipe_func_graph_widget = PipeFuncGraphWidget(dot_source=dot_string)
    reset_button = widgets.Button(description="Reset Zoom")
    direction_selector = widgets.Dropdown(
        options=["bidirectional", "downstream", "upstream", "single"],
        value="bidirectional",
        description="Direction:",
    )
    search_input = widgets.Text(
        value="",
        placeholder="Search...",
        description="Search:",
        disabled=False,
    )

    # Define button actions
    def reset_graph(_: Any) -> None:
        pipe_func_graph_widget.send({"action": "reset_zoom"})

    def update_direction(change: dict) -> None:
        pipe_func_graph_widget.selected_direction = change["new"]

    def perform_search(change: dict) -> None:
        pipe_func_graph_widget.send({"action": "search", "query": change["new"]})

    reset_button.on_click(reset_graph)
    direction_selector.observe(update_direction, names="value")
    search_input.observe(perform_search, names="value")

    # Display widgets
    return widgets.VBox(
        [
            widgets.HBox([reset_button, direction_selector, search_input]),
            pipe_func_graph_widget,
        ],
    )
