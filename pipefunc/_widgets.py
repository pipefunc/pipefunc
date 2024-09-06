from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipefunc import Pipeline

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from pipefunc import Pipeline


class PipelineWidget:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

        # Initializing Widgets
        self.info_button = widgets.Button(description="Show Pipeline Info")
        self.visualize_button = widgets.Button(description="Visualize Pipeline")
        self.function_selector = widgets.Dropdown(
            options=[func.__name__ for func in self.pipeline.functions],
            description="Choose Function:",
        )
        self.input_output_display = widgets.Output()
        self.function_info_display = widgets.Output()

        # Layout and Organization
        self.actions_box = widgets.HBox([self.info_button, self.visualize_button])
        self.main_box = widgets.VBox(
            [
                self.actions_box,
                self.input_output_display,
                self.function_selector,
                self.function_info_display,
            ],
        )

        # Event Handlers
        self.info_button.on_click(self.show_pipeline_info)
        self.visualize_button.on_click(self.visualize_pipeline)
        self.function_selector.observe(self.show_function_details, "value")

    def show_pipeline_info(self, b=None):
        """Displays pipeline parameters and types for all functions."""
        with self.input_output_display:
            clear_output()
            print("Pipeline Parameters and Types:")
            for func in self.pipeline.functions:
                print(f"\nFunction: {func.__name__}")
                for param, typ in func.parameter_annotations.items():
                    print(f"  - {param}: {typ.__name__}")

                for output_name, output_type in func.output_annotation.items():
                    print(f"Output: {output_name} -> {output_type.__name__}")

    def visualize_pipeline(self, b=None):
        """Visualizes the pipeline."""
        with self.input_output_display:
            clear_output()
            plt.clf()  # Clear the current figure
            print("Pipeline Visualization:")
            self.pipeline.visualize()
            plt.show()

    def show_function_details(self, change):
        """Displays details about the selected function."""
        selected_func_name = change.new
        if selected_func_name:
            func = next(f for f in self.pipeline.functions if f.__name__ == selected_func_name)
            with self.function_info_display:
                clear_output()
                print(f"Function: {func.__name__}")
                # Handle output annotation, which may be a dict or a single value
                output_annotation = func.output_annotation
                if isinstance(output_annotation, dict):
                    for output_name, output_type in output_annotation.items():
                        print(f"Output: {output_name} -> {output_type.__name__}")
                else:
                    print(f"Output: {func.output_name} -> {output_annotation.__name__}")

                for param, typ in func.parameter_annotations.items():
                    print(f"  - {param}: {typ.__name__}")

    def display(self):
        """Displays the widget in the notebook."""
        display(self.main_box)
