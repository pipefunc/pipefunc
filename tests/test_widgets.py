"""Test the PipelineWidget class."""

from unittest.mock import MagicMock

import ipywidgets as widgets
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc._widgets import PipelineWidget


@pytest.fixture
def mock_pipeline():
    """Mock the Pipeline object."""
    return MagicMock()


@pytest.fixture
def pipeline():
    """Create a real Pipeline object."""

    @pipefunc
    def add_one(x):
        return x + 1

    @pipefunc
    def multiply_by_two(x):
        return x * 2

    return Pipeline([add_one, multiply_by_two])


def test_initial_widget_state(mock_pipeline):
    """Test initial state of the widget."""
    widget = PipelineWidget(pipeline=mock_pipeline)
    assert isinstance(widget.info_button, widgets.Button)
    assert widget.info_button.description == "Show Pipeline Info"

    assert isinstance(widget.visualize_button, widgets.Button)
    assert widget.visualize_button.description == "Visualize Pipeline"

    assert isinstance(widget.visualize_type, widgets.Dropdown)
    assert widget.visualize_type.value == "matplotlib"

    assert isinstance(widget.fig_width, widgets.IntSlider)
    assert widget.fig_width.value == 10

    assert isinstance(widget.fig_height, widgets.IntSlider)
    assert widget.fig_height.value == 10


def test_show_pipeline_info_calls(mock_pipeline):
    """Test the _show_pipeline_info method to ensure it interacts with pipeline correctly."""
    widget = PipelineWidget(pipeline=mock_pipeline)
    widget._show_pipeline_info()
    # Check the methods that should be called on the pipeline
    # You have to adapt this part according to what the mock pipeline should do
    # Add assertions here based on expected interactions


def test_visualize_pipeline_matplotlib(mock_pipeline):
    """Test the visualize pipeline functionality for matplotlib."""
    widget = PipelineWidget(pipeline=mock_pipeline)
    widget.visualize_type.value = "matplotlib"
    widget.fig_width.value = 15
    widget.fig_height.value = 10

    widget._visualize_pipeline()
    # Check that the appropriate functions on the pipeline are called
    # for visualization using matplotlib
    mock_pipeline.visualize.assert_called_with(figsize=(15, 10))


def test_visualize_pipeline_holoviews(pipeline: Pipeline):
    """Test the visualize pipeline functionality for HoloViews."""
    widget = PipelineWidget(pipeline=pipeline)
    widget.visualize_type.value = "holoviews"
    widget._visualize_pipeline()


def test_on_size_change(mock_pipeline):
    """Test the size change triggers re-visualization."""
    widget = PipelineWidget(pipeline=mock_pipeline)

    # Set initial visualization type to "matplotlib"
    widget.visualize_type.value = "matplotlib"
    # Manually call the pipeline visualization to check if re-visualization occurs.
    widget._visualize_pipeline()

    # Adjust the width and height sliders to trigger re-visualization
    widget.fig_width.value = 20
    widget.fig_height.value = 15

    # This should trigger automatic re-visualization due to observers
    widget._on_size_change({"name": "value", "new": 20, "type": "change"})

    # Check if the mock pipeline's visualize method was called
    mock_pipeline.visualize.assert_called_with(figsize=(20, 15))


def test_on_visualize_type_change(mock_pipeline):
    """Test changing the visualization type."""
    widget = PipelineWidget(pipeline=mock_pipeline)

    # Assume the initial type is "matplotlib".
    widget.visualize_type.value = "matplotlib"

    # Change to "holoviews" and test behavior.
    widget._on_visualize_type_change({"name": "value", "old": "matplotlib", "new": "holoviews"})
    assert widget.fig_width.layout.display == "none"
    assert widget.fig_height.layout.display == "none"

    # Change back to "matplotlib" and test behavior.
    widget._on_visualize_type_change({"name": "value", "old": "holoviews", "new": "matplotlib"})
    assert widget.fig_width.layout.display == ""
    assert widget.fig_height.layout.display == ""
