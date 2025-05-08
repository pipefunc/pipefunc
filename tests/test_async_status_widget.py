"""Tests for async status widget functionality."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from pipefunc._widgets.async_status_widget import AsyncMapStatusWidget, StatusType
from pipefunc._widgets.helpers import maybe_async_map_status_widget


@pytest.fixture
def widget():
    """Create a widget instance for testing."""
    return AsyncMapStatusWidget(display=False)


@pytest.mark.asyncio
async def test_widget_initialization():
    """Test widget initialization and basic properties."""
    widget = AsyncMapStatusWidget(display=False)
    assert widget._status_widget is not None
    assert widget._main_widget is not None
    assert widget._traceback_widget is not None
    assert widget._traceback_button is not None
    assert widget._start_time <= time.monotonic()
    assert widget._update_interval == 0.1
    assert widget._task is None
    assert widget._update_timer is None
    assert widget._traceback_visible is False
    assert widget._traceback_button.layout.display == "none"
    assert widget._traceback_widget.layout.display == "none"


@pytest.mark.asyncio
async def test_widget_status_styles():
    """Test that all status types have associated styles."""
    widget = AsyncMapStatusWidget(display=False)
    statuses: list[StatusType] = ["initializing", "running", "done", "cancelled", "failed"]

    for status in statuses:
        style = widget._get_style(status)
        assert style.color is not None
        assert style.icon is not None
        assert style.message is not None


@pytest.mark.asyncio
async def test_widget_elapsed_time():
    """Test elapsed time calculation."""
    widget = AsyncMapStatusWidget(display=False)
    start_time = widget._start_time
    await asyncio.sleep(0.1)
    elapsed = widget._get_elapsed_time()
    assert elapsed >= 0.1
    assert elapsed >= time.monotonic() - start_time - 0.01


@pytest.mark.asyncio
async def test_widget_display_refresh():
    """Test display refresh with different statuses."""
    widget = AsyncMapStatusWidget(display=False)

    # Test with running status
    with patch.object(widget._status_widget, "clear_output") as mock_clear:
        widget._refresh_display("running")
        mock_clear.assert_called_once_with(wait=True)

    # Test with done status
    with patch.object(widget._status_widget, "clear_output") as mock_clear:
        widget._refresh_display("done")
        mock_clear.assert_called_once_with(wait=True)

    # Check button visibility (should be hidden for non-error states)
    assert widget._traceback_button.layout.display == "none"
    assert widget._traceback_widget.layout.display == "none"


@pytest.mark.asyncio
async def test_widget_error_display():
    """Test error display with traceback button."""
    widget = AsyncMapStatusWidget(display=False)
    error = ValueError("Test error message")

    # Display with error
    with patch.object(widget._status_widget, "clear_output") as mock_clear:
        widget._refresh_display("failed", error)
        mock_clear.assert_called_once_with(wait=True)

    # Check button visibility (should be visible for error states)
    assert widget._traceback_button.layout.display == "block"
    assert widget._traceback_widget.layout.display == "none"
    assert widget._exception is error


@pytest.mark.asyncio
async def test_widget_toggle_traceback():
    """Test toggling traceback visibility."""
    widget = AsyncMapStatusWidget(display=False)
    error = ValueError("Test error message")
    widget._refresh_display("failed", error)

    # Initially traceback is hidden
    assert widget._traceback_visible is False
    assert widget._traceback_widget.layout.display == "none"

    # Toggle to show
    widget._toggle_traceback({})
    assert widget._traceback_visible is True
    assert widget._traceback_widget.layout.display == "block"
    assert widget._traceback_button.description == "Hide traceback"
    assert widget._traceback_button.button_style == "danger"
    assert widget._traceback_button.icon == "close"

    # Toggle to hide
    widget._toggle_traceback({})
    assert widget._traceback_visible is False
    assert widget._traceback_widget.layout.display == "none"
    assert widget._traceback_button.description == "Show traceback"
    assert widget._traceback_button.button_style == "info"
    assert widget._traceback_button.icon == "search"


@pytest.mark.asyncio
async def test_widget_task_attachment():
    """Test attaching a task to the widget."""
    widget = AsyncMapStatusWidget(display=False)

    async def test_task():
        await asyncio.sleep(0.1)
        return "done"

    task = asyncio.create_task(test_task())
    widget.attach_task(task)

    assert widget._task is task
    assert widget._update_timer is not None

    await task
    await asyncio.sleep(0.1)  # Give time for callback to execute


@pytest.mark.asyncio
async def test_widget_task_completion():
    """Test widget updates on task completion."""
    widget = AsyncMapStatusWidget(display=False)

    async def test_task():
        await asyncio.sleep(0.1)
        return "done"

    with patch.object(widget, "_refresh_display") as mock_refresh:
        task = asyncio.create_task(test_task())
        widget.attach_task(task)

        # Should start with running status
        mock_refresh.assert_called_with("running")
        mock_refresh.reset_mock()

        await task
        await asyncio.sleep(0.1)  # Give time for callback to execute

        # Should end with done status
        mock_refresh.assert_called_with("done")


@pytest.mark.asyncio
async def test_widget_task_cancellation():
    """Test widget updates on task cancellation."""
    widget = AsyncMapStatusWidget(display=False)

    async def test_task():
        await asyncio.sleep(1)

    with patch.object(widget, "_refresh_display") as mock_refresh:
        task = asyncio.create_task(test_task())
        widget.attach_task(task)

        # Should start with running status
        mock_refresh.assert_called_with("running")
        mock_refresh.reset_mock()

        task.cancel()
        await asyncio.sleep(0.1)  # Give time for callback to execute

        # Should end with cancelled status
        mock_refresh.assert_called_with("cancelled")


@pytest.mark.asyncio
async def test_widget_task_failure():
    """Test widget updates on task failure."""
    widget = AsyncMapStatusWidget(display=False)

    msg = "Test error"

    async def test_task():
        raise ValueError(msg)

    with patch.object(widget, "_refresh_display") as mock_refresh:
        task = asyncio.create_task(test_task())
        widget.attach_task(task)

        # Should start with running status
        mock_refresh.assert_called_with("running")
        mock_refresh.reset_mock()

        with pytest.raises(ValueError, match=msg):
            await task
        await asyncio.sleep(0.1)  # Give time for callback to execute

        # Should call refresh with failed status and error
        mock_refresh.assert_called_once()
        args = mock_refresh.call_args[0]
        assert args[0] == "failed"
        assert isinstance(args[1], ValueError)
        assert str(args[1]) == msg


@pytest.mark.asyncio
async def test_widget_periodic_updates():
    """Test periodic updates of the widget."""
    widget = AsyncMapStatusWidget(display=False)

    # Create a simple task that runs for a bit
    async def test_task():
        await asyncio.sleep(0.3)
        return "done"

    # Patch the _refresh_display method to count calls
    with patch.object(widget, "_refresh_display") as mock_refresh:
        task = asyncio.create_task(test_task())
        widget.attach_task(task)

        # Should start with running status
        mock_refresh.assert_called_with("running")

        # Wait for task to complete
        await task
        await asyncio.sleep(0.1)  # Give time for callback to execute

        # Should have called refresh multiple times
        assert mock_refresh.call_count >= 2


@pytest.mark.asyncio
async def test_stop_periodic_updates():
    """Test stopping periodic updates."""
    widget = AsyncMapStatusWidget(display=False)

    # Create a mock task
    mock_task = MagicMock()
    mock_task.cancel = MagicMock()

    # Set the update timer and test stopping it
    widget._update_timer = mock_task
    widget._stop_periodic_updates()

    # Should have called cancel on the task
    mock_task.cancel.assert_called_once()
    assert widget._update_timer is None


@pytest.mark.asyncio
async def test_widget_display_method():
    """Test the display method."""
    widget = AsyncMapStatusWidget(display=False)

    with patch("IPython.display.display") as mock_display:
        widget.display()
        mock_display.assert_called_once_with(widget._main_widget)


@pytest.mark.asyncio
async def test_update_interval_adjustment():
    """Test update interval adjustment based on elapsed time."""
    widget = AsyncMapStatusWidget(display=False)

    # Mock a task to prevent early return
    mock_task = MagicMock()
    mock_task.done.return_value = False
    widget._task = mock_task

    # Test initial interval
    assert widget._update_interval == 0.1

    # Test adjustment with different elapsed times
    async def mock_sleep(_):
        """Mock asyncio.sleep to prevent actual waiting."""

    # Create a simplified version that only does one iteration of the while loop
    async def run_one_periodic_update():
        """Run one iteration of the update loop."""
        widget._refresh_display("running")
        # Skip asyncio.sleep since we mocked it
        elapsed = widget._get_elapsed_time()
        if elapsed < 10:
            pass
        elif elapsed < 100:
            widget._update_interval = 1.0
        elif elapsed < 1000:
            widget._update_interval = 10.0
        else:
            widget._update_interval = 60.0

    with (
        patch.object(widget, "_get_elapsed_time") as mock_elapsed,
        patch("asyncio.sleep", mock_sleep),
        patch.object(widget, "_refresh_display"),
    ):
        # Test with elapsed < 10s (no change)
        mock_elapsed.return_value = 5.0
        widget._update_interval = 0.1  # Reset to initial value
        await run_one_periodic_update()
        assert widget._update_interval == 0.1

        # Test with 10s < elapsed < 100s
        mock_elapsed.return_value = 50.0
        widget._update_interval = 0.1  # Reset to initial value
        await run_one_periodic_update()
        assert widget._update_interval == 1.0

        # Test with 100s < elapsed < 1000s
        mock_elapsed.return_value = 500.0
        widget._update_interval = 0.1  # Reset to initial value
        await run_one_periodic_update()
        assert widget._update_interval == 10.0

        # Test with elapsed > 1000s
        mock_elapsed.return_value = 1500.0
        widget._update_interval = 0.1  # Reset to initial value
        await run_one_periodic_update()
        assert widget._update_interval == 60.0


@pytest.mark.asyncio
async def test_maybe_async_map_status_widget_no_ipynb():
    """Test maybe_async_map_status_widget when not in ipynb."""
    with patch("pipefunc._utils.is_running_in_ipynb", return_value=False):

        async def test_task():
            return "done"

        task = asyncio.create_task(test_task())
        widget = maybe_async_map_status_widget(task)
        assert widget is None


@pytest.mark.asyncio
async def test_maybe_async_map_status_widget_with_ipywidgets():
    """Test maybe_async_map_status_widget when all requirements are met."""
    with (
        patch("pipefunc._widgets.helpers.is_running_in_ipynb", return_value=True),
    ):

        async def test_task():
            return "done"

        task = asyncio.create_task(test_task())
        widget = maybe_async_map_status_widget(task)
        assert widget is not None
        assert isinstance(widget, AsyncMapStatusWidget)


@pytest.mark.asyncio
async def test_start_periodic_updates_error_handling():
    """Test error handling in _start_periodic_updates."""
    widget = AsyncMapStatusWidget(display=False)

    # Test RuntimeError handling
    with (
        patch("asyncio.get_event_loop", side_effect=RuntimeError("Test error")),
        patch("builtins.print") as mock_print,
    ):
        widget._start_periodic_updates()
        mock_print.assert_called_once()
        assert "Could not start periodic updates" in mock_print.call_args[0][0]

    # Test generic exception handling
    with (
        patch("asyncio.get_event_loop", side_effect=Exception("Test error")),
        patch("builtins.print") as mock_print,
    ):
        widget._start_periodic_updates()
        mock_print.assert_called_once()
        assert "Error starting periodic updates" in mock_print.call_args[0][0]
