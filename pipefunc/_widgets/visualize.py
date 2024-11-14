from typing import Any

import ipywidgets as widgets

from .graphviz_widget import PipeFuncGraphWidget


def graph_widget(dot_string: str = "digraph { a -> b; b -> c; c -> a; }") -> widgets.VBox:
    pipe_func_graph_widget = PipeFuncGraphWidget(dot_source=dot_string)
    reset_button = widgets.Button(description="Reset Zoom")
    direction_selector = widgets.Dropdown(
        options=["bidirectional", "downstream", "upstream", "single"],
        value="bidirectional",
        description="Direction:",
    )
    search_input = widgets.Text(
        placeholder="Search...",
        description="Search:",
    )
    search_type_selector = widgets.Dropdown(
        options=["exact", "included", "regex"],
        value="exact",
        description="Search Type:",
    )
    case_toggle = widgets.ToggleButton(
        value=False,
        description="Case Sensitive",
        icon="check",
    )

    # Define button actions
    def reset_graph(_: Any) -> None:
        pipe_func_graph_widget.send({"action": "reset_zoom"})

    def update_direction(change: dict) -> None:
        pipe_func_graph_widget.selected_direction = change["new"]

    def perform_search(change: dict) -> None:
        pipe_func_graph_widget.send({"action": "search", "query": change["new"]})

    def update_search_type(change: dict) -> None:
        pipe_func_graph_widget.search_type = change["new"]

    def toggle_case_sensitive(change: dict) -> None:
        pipe_func_graph_widget.case_sensitive = change["new"]

    reset_button.on_click(reset_graph)
    direction_selector.observe(update_direction, names="value")
    search_input.observe(perform_search, names="value")
    search_type_selector.observe(update_search_type, names="value")
    case_toggle.observe(toggle_case_sensitive, names="value")

    # Display widgets
    return widgets.VBox(
        [
            widgets.HBox([reset_button, direction_selector]),
            widgets.HBox([search_input, search_type_selector, case_toggle]),
            pipe_func_graph_widget,
        ],
    )
