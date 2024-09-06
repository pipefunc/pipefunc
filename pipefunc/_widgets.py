import ipywidgets as widgets
from IPython.display import display

from pipefunc import Pipeline

CNT = {"count": 0}


class PipelineWidget:
    def __init__(self, pipeline: Pipeline):
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

        # Widgets to control figure size
        self.fig_width = widgets.IntSlider(
            value=5,
            min=1,
            max=20,
            step=1,
            description="Width:",
        )
        self.fig_height = widgets.IntSlider(
            value=4,
            min=1,
            max=20,
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

    def create_info_tab(self):
        """Creates the info tab layout."""
        return widgets.VBox(
            [
                widgets.HTML("<h2>Pipeline Information</h2>"),
                self.info_button,
                widgets.HTML("<hr>"),
                self.info_output_display,
            ]
        )

    def create_visualize_tab(self):
        """Creates the visualize tab layout."""
        return widgets.VBox(
            [
                widgets.HTML("<h2>Visualize Pipeline</h2>"),
                self.fig_width,
                self.fig_height,
                widgets.HTML("<hr>"),
                self.visualize_button,
                self.visualize_output_display,
            ]
        )

    def show_pipeline_info(self, b=None):
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

    def visualize_pipeline(self, b=None):
        """Visualizes the pipeline with the specified figure size."""
        with self.visualize_output_display:
            self.visualize_output_display.clear_output(wait=True)

            # Get the figure size from the widgets
            figsize = (self.fig_width.value, self.fig_height.value)

            display(widgets.HTML("<h3>Pipeline Visualization</h3>"))
            self.pipeline.visualize(figsize=figsize)  # Pass the figsize to visualize function

    def on_size_change(self, change):
        """Called when the figure size sliders are updated. Automatically re-renders the visualization."""
        self.visualize_pipeline()  # Automatically re-visualize pipeline when slider values change.

    def display(self):
        """Displays the widget in the notebook."""
        display(self.tab)
