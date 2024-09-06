from typing import Any

import ipywidgets as widgets
from IPython.display import display

from pipefunc import PipeFunc, Pipeline


class PipelineWidget:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

        # Initializing Widgets with styles
        self.info_button = widgets.Button(
            description="Show Pipeline Info",
            button_style="info",
            tooltip="Click to view pipeline information",
        )
        self.visualize_button = widgets.Button(
            description="Visualize Pipeline",
            button_style="primary",
            tooltip="Click to visualize the pipeline",
        )

        # Dropdown for choosing visualization type
        self.visualize_type = widgets.Dropdown(
            options=["matplotlib", "holoviews"],
            value="matplotlib",
            description="Type:",
        )

        # Widgets to control figure size (only relevant for matplotlib)
        self.fig_width = widgets.IntSlider(
            value=10,
            min=1,
            max=50,
            step=1,
            description="Width:",
        )
        self.fig_height = widgets.IntSlider(
            value=10,
            min=1,
            max=50,
            step=1,
            description="Height:",
        )

        # Output areas for each tab
        self.info_output_display = widgets.Output()
        self.visualize_output_display = widgets.Output()

        # Tab setup
        self.tab = widgets.Tab()
        self.tab.children = [self.create_info_tab(), self.create_visualize_tab()]
        self.tab.set_title(0, "Info")
        self.tab.set_title(1, "Visualize")

        # Bind the button click events
        self.info_button.on_click(self.show_pipeline_info)
        self.visualize_button.on_click(self.visualize_pipeline)

        # Bind the slider change events
        self.fig_width.observe(self.on_size_change, names="value")
        self.fig_height.observe(self.on_size_change, names="value")

        # Bind the dropdown change event
        self.visualize_type.observe(self.on_visualize_type_change, names="value")

    def create_info_tab(self) -> widgets.VBox:
        """Creates the info tab layout."""
        return widgets.VBox(
            [
                widgets.HTML("<h2>Pipeline Information</h2>"),
                self.info_button,
                widgets.HTML("<hr>"),
                self.info_output_display,
            ],
        )

    def create_visualize_tab(self) -> widgets.VBox:
        """Creates the visualize tab layout."""
        return widgets.VBox(
            [
                widgets.HTML("<h2>Visualize Pipeline</h2>"),
                self.visualize_type,  # Dropdown to select visualization type
                self.fig_width,  # Sliders for size, potentially hidden
                self.fig_height,  # Sliders for size, potentially hidden
                widgets.HTML("<hr>"),
                self.visualize_button,
                self.visualize_output_display,
            ],
        )

    def show_pipeline_info(self, _button: widgets.Button | None = None) -> None:
        """Displays pipeline parameters and types using the factored-out helper functions."""
        with self.info_output_display:
            self.info_output_display.clear_output(wait=True)

            # Root arguments and defaults from the pipeline
            root_args = set(self.pipeline.topological_generations.root_args)
            defaults = self.pipeline.defaults

            # Start the HTML content, including the legend
            html_content = f"""
                {_build_legend()}
            """

            # Add parameter table for each function
            for func in self.pipeline.functions:
                html_content += f"<h4><b>Function:</b> {_format_code(func.__name__)}</h4>"

                # Prepare headers for the parameter table
                headers = ["Parameter", "Type", "Default Value", "Returned By"]

                # Build the rows for the parameter table
                rows = [
                    _format_param(
                        param,
                        param_type=func.parameter_annotations.get(param, "—"),
                        is_root=param in root_args,
                        default_value=defaults.get(param),
                        returned_by=self.pipeline.output_to_func.get(param, "—"),
                    )
                    for param in func.parameters
                ]

                # Generate the parameter table for this function
                html_content += _build_table(headers, rows)
                # Outputs of the function (if any)
                if func.output_annotation:
                    html_content += "<h4>Outputs</h4><ul>"
                    for output_name, output_type in func.output_annotation.items():
                        html_content += f"<li class='output-type'><b>{output_name}</b>: {output_type.__name__}</li>"
                    html_content += "</ul>"
                html_content += "<hr>"

            # Single combined table for the PipeFunc attribute information
            html_content += "<h3>PipeFunc Attributes Overview</h3>"
            html_content += _build_combined_pipefunc_info_table(self.pipeline.functions)

            # Display the resulting content
            display(widgets.HTML(html_content))

    def visualize_pipeline(self, _button: widgets.Button | None = None) -> None:
        """Visualizes the pipeline with the selected method and size."""
        if self.visualize_type.value == "holoviews":
            import holoviews as hv
            import panel as pn

            hv.extension("bokeh")

        with self.visualize_output_display:
            self.visualize_output_display.clear_output(wait=True)

            # Determine which visualization method to use
            if self.visualize_type.value == "matplotlib":
                figsize = (self.fig_width.value, self.fig_height.value)
                display(widgets.HTML("<h3>Pipeline Visualization (Matplotlib)</h3>"))
                self.pipeline.visualize(figsize=figsize)  # Pass the figsize to visualize function

            elif self.visualize_type.value == "holoviews":
                display(widgets.HTML("<h3>Pipeline Visualization (HoloViews)</h3>"))
                hv_output = self.pipeline.visualize_holoviews()
                display(pn.ipywidget(hv_output))  # Render the HoloViews object

    def on_size_change(self, _change: dict) -> None:
        """Called when the figure size sliders are updated. Automatically re-renders the visualization."""
        if self.visualize_type.value == "matplotlib":
            self.visualize_pipeline()  # Automatically re-visualize pipeline when slider values change.

    def on_visualize_type_change(self, change: dict) -> None:
        """Called when the visualization type is changed. Shows/hides the size sliders."""
        if change["new"] == "matplotlib":
            self.fig_width.layout.display = ""  # Make the sliders visible
            self.fig_height.layout.display = ""  # Make the sliders visible
            self.visualize_pipeline()  # Trigger visualization with size control
        else:  # For HoloViews
            self.fig_width.layout.display = "none"  # Hide the sliders for HoloViews
            self.fig_height.layout.display = "none"  # Hide the sliders for HoloViews
            self.visualize_pipeline()  # Trigger the chosen visualization method

    def display(self) -> None:
        """Displays the widget in the notebook."""
        display(self.tab)


def _build_table(headers: list[str], rows: list[list[str]]) -> str:
    # Build the header row
    header_html = _build_row(headers, cell_tag="th")

    # Build each row, joining them together
    rows_html = "".join([_build_row(row) for row in rows])

    # Wrap everything in <table> tags with overall table styling
    return f"""
        <table>
            <thead>{header_html}</thead>
            <tbody>{rows_html}</tbody>
        </table>
    """


def _build_row(cells: list[str], cell_tag: str = "td") -> str:
    return f"<tr>{''.join([f'<{cell_tag}>{cell}</{cell_tag}>' for cell in cells])}</tr>"


def _format_param(
    param: str,
    *,
    param_type: Any,
    is_root: bool,
    default_value: str | None,
    returned_by: PipeFunc | str,
) -> list[str]:
    # Format the parameter name, bold and red if it's a root argument
    param_html = (
        f"<span style='color: #e74c3c; font-weight: bold;'>{param}</span>" if is_root else param
    )

    # Format the default value, green if it exists
    if default_value is not None:
        default_html = f"<span style='color: #2ecc71;'>{default_value}</span>"
    else:
        default_html = "—"

    if isinstance(returned_by, PipeFunc):
        returned_by = _format_code(returned_by.__name__)

    if param_type != "—":
        param_type = _format_code(param_type.__name__)

    return [param_html, param_type, default_html, returned_by]


def _build_legend() -> str:
    return """
        <div class="legend" style="font-size: 14px; margin-bottom: 15px;">
            <b>Legend:</b>
            <ul>
                <li><span style="color: #e74c3c; font-weight: bold;">Bold & Red</span> parameters are pipeline root arguments (root_args).</li>
                <li>Parameters with <span style="color: #2ecc71;">green text</span> have default values.</li>
            </ul>
        </div>
    """


def _build_combined_pipefunc_info_table(functions: list[PipeFunc]) -> str:
    """Build a combined table of functions with each function as a column."""
    headers = ["Attribute"] + [func.__name__ for func in functions]

    # Attributes to be displayed
    attributes = ["Cache", "Profile", "MapSpec", "Resources", "Debug"]

    # Collect the value rows for each function
    rows = []
    for attribute in attributes:
        row = [attribute]
        for func in functions:
            # Get attribute based on the name in the attributes list
            value = getattr(func, attribute.lower(), None)
            if isinstance(value, bool):
                # For basic types: convert to string directly
                row.append("✅" if value else "❌")
            elif isinstance(value, int | str):
                # For basic types: convert to string directly
                row.append(str(value))
            elif value is None:
                # handle None-values
                row.append("—")
            else:
                # handle special cases (like MapSpec, Resources, etc.)
                row.append(str(value) if value is not None else "—")
        rows.append(row)

    # Generate the table HTML by plugging in the headers and rows
    return _build_table(headers, rows)


def _format_code(text: str) -> str:
    return f"<code>{text}</code>"
