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
        """Displays pipeline parameters and types for all functions."""
        with self.info_output_display:
            self.info_output_display.clear_output(wait=True)

            html_content = "<h3>Pipeline Parameters and Types</h3>"
            for func in self.pipeline.functions:
                html_content += f"<b>Function:</b> {func.__name__}<br>"
                for param, typ in func.parameter_annotations.items():
                    html_content += f"  - <i>{param}:</i> {typ.__name__}<br>"
                for output_name, output_type in func.output_annotation.items():
                    html_content += f"<b>Output:</b> {output_name} -> {output_type.__name__}<br>"
                html_content += "<hr>"

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
