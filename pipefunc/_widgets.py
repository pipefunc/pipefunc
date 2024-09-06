import ipywidgets as widgets
from IPython.display import display

from pipefunc import Pipeline


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
        """Displays pipeline parameters and types for all functions in a structured and visually attractive way."""
        with self.info_output_display:
            self.info_output_display.clear_output(wait=True)

            # Root arguments and defaults from the pipeline
            root_args = set(self.pipeline.topological_generations.root_args)
            defaults = self.pipeline.defaults

            # Starting HTML content with embedded CSS for styling
            html_content = """
                <style>
                    .pipeline-info h3 {
                        font-family: Arial, Helvetica, sans-serif;
                        color: #2c3e50;
                        text-align: center;
                    }
                    .pipeline-info table {
                        width: 100%;
                        border-collapse: collapse;
                        margin: 15px 0;
                        font-family: Arial, sans-serif;
                        font-size: 14px;
                        text-align: left;
                    }
                    .pipeline-info th, .pipeline-info td {
                        padding: 8px 12px;
                        border: 1px solid #dee2e6;
                    }
                    .pipeline-info th {
                        background-color: #3498db;
                        color: white;
                    }
                    .pipeline-info td.key-root {
                        font-weight: bold;
                        color: #e74c3c;
                    }
                    .pipeline-info td.default {
                        color: #2ecc71;
                    }
                    .pipeline-info hr {
                        margin: 25px 0;
                    }
                    .pipeline-info .output {
                        color: #8e44ad;
                    }
                </style>
                <div class="pipeline-info">
                <h3>Pipeline Parameters and Types</h3>
            """

            # Iterate through functions in the pipeline
            for func in self.pipeline.functions:
                html_content += f"<h4><b>Function:</b> {func.__name__}</h4>"

                html_content += """
                <table>
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Type</th>
                            <th>Default Value</th>
                        </tr>
                    </thead>
                    <tbody>
                """

                # For each function, go through its parameters
                for param in func.parameters:
                    is_root_arg = param in root_args
                    param_type = func.parameter_annotations.get(
                        param,
                        "Any",
                    )  # Fallback to 'Any' if no type
                    default_value = defaults.get(param, None)

                    # Define CSS classes for custom styling
                    param_class = "key-root" if is_root_arg else ""
                    default_display = (
                        f"<span class='default'>{default_value}</span>"
                        if default_value is not None
                        else "—"
                    )

                    # Add the table row with the parameter, type, and default value
                    html_content += f"""
                    <tr>
                        <td class='{param_class}'>{param}</td>
                        <td>{param_type}</td>
                        <td>{default_display}</td>
                    </tr>
                    """

                html_content += "</tbody></table>"

                # Outputs of the function
                if func.output_annotation:
                    html_content += "<h4>Outputs</h4><ul>"
                    for output_name, output_type in func.output_annotation.items():
                        html_content += (
                            f"<li class='output'><b>{output_name}</b>: {output_type.__name__}</li>"
                        )
                    html_content += "</ul>"

                # Divider to separate sections
                html_content += "<hr>"

            # End of the HTML div
            html_content += "</div>"

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
