"""Tests for async status widget functionality."""

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from pipefunc._widgets.async_status_widget import AsyncTaskStatusWidget, StatusType
from pipefunc._widgets.helpers import maybe_async_task_status_widget


@pytest.fixture
def widget():
    """Create a widget instance for testing."""
    return AsyncTaskStatusWidget(display=False)


@pytest.mark.asyncio
async def test_widget_initialization():
    """Test widget initialization and basic properties."""
    widget = AsyncTaskStatusWidget(display=False)
    assert widget._status_html_widget is not None
    assert widget.widget is not None
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
async def test_widget_status_styles() -> None:
    """Test that all status types have associated styles."""
    widget = AsyncTaskStatusWidget(display=False)
    statuses: list[StatusType] = ["initializing", "running", "done", "cancelled", "failed"]

    for status in statuses:
        style = widget._get_style(status)
        assert style.color is not None
        assert style.icon is not None
        assert style.message is not None


@pytest.mark.asyncio
async def test_widget_elapsed_time():
    """Test elapsed time calculation."""
    widget = AsyncTaskStatusWidget(display=False)
    start_time = widget._start_time
    await asyncio.sleep(0.1)
    elapsed = widget._get_elapsed_time()
    assert elapsed >= 0.1
    assert elapsed >= time.monotonic() - start_time - 0.01


@pytest.mark.asyncio
async def test_widget_display_refresh():
    """Test display refresh with different statuses."""
    widget = AsyncTaskStatusWidget(display=False)

    # Test with running status
    with patch.object(
        type(widget._status_html_widget),
        "value",
        new_callable=PropertyMock,
    ) as mock_value_setter:
        widget._refresh_display("running")
        mock_value_setter.assert_called_once()

    # Test with done status
    with patch.object(
        type(widget._status_html_widget),
        "value",
        new_callable=PropertyMock,
    ) as mock_value_setter:
        widget._refresh_display("done")
        mock_value_setter.assert_called_once()

    # Check button visibility (should be hidden for non-error states)
    assert widget._traceback_button.layout.display == "none"
    assert widget._traceback_widget.layout.display == "none"


@pytest.mark.asyncio
async def test_widget_error_display():
    """Test error display with traceback button."""
    widget = AsyncTaskStatusWidget(display=False)
    error = ValueError("Test error message")

    # Display with error
    with (
        patch.object(
            type(widget._status_html_widget),
            "value",
            new_callable=PropertyMock,
        ) as mock_status_value,
    ):
        widget._refresh_display("failed", error)
        mock_status_value.assert_called_once()

    # Check button visibility (should be visible for error states)
    assert widget._traceback_button.layout.display == "block"
    assert widget._traceback_widget.layout.display == "none"
    assert widget._exception is error


@pytest.mark.asyncio
async def test_widget_toggle_traceback():
    """Test toggling traceback visibility."""
    widget = AsyncTaskStatusWidget(display=False)
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
    widget = AsyncTaskStatusWidget(display=False)

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
    widget = AsyncTaskStatusWidget(display=False)

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
    widget = AsyncTaskStatusWidget(display=False)

    async def test_task():
        await asyncio.sleep(1)

    with patch.object(widget, "_refresh_display") as mock_refresh:
        task = asyncio.create_task(test_task())
        widget.attach_task(task)

        # Should start with running status
        mock_refresh.assert_called_with("running")
        mock_refresh.reset_mock()

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task  # Allow task to process cancellation
        await asyncio.sleep(0.01)  # Give time for done_callback to execute

        # Should end with cancelled status
        mock_refresh.assert_called_with("cancelled")


@pytest.mark.asyncio
async def test_widget_task_failure():
    """Test widget updates on task failure."""
    widget = AsyncTaskStatusWidget(display=False)

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
    widget = AsyncTaskStatusWidget(display=False)

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
    widget = AsyncTaskStatusWidget(display=False)

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
    widget = AsyncTaskStatusWidget(display=False)

    with patch("IPython.display.display") as mock_display:
        widget.display()
        mock_display.assert_called_once_with(widget.widget)


@pytest.mark.asyncio
async def test_update_interval_adjustment():
    """Test update interval adjustment based on elapsed time."""
    widget = AsyncTaskStatusWidget(display=False)

    # Create a mock task that never finishes
    mock_task = MagicMock()
    mock_task.done.return_value = False
    widget._task = mock_task

    # Test initial interval
    assert widget._update_interval == 0.1

    # Define a function to check interval adjustment with a specific elapsed time
    async def test_interval_with_elapsed_time(elapsed_time, expected_interval):
        """Test interval adjustment for a specific elapsed time."""
        # Reset interval to initial value
        widget._update_interval = 0.1

        # Make the first while loop iteration run once and then exit
        done_after_one_iteration = [False]

        def mock_done():
            if done_after_one_iteration[0]:
                return True
            done_after_one_iteration[0] = True
            return False

        mock_task.done = mock_done

        # Mock elapsed time and asyncio.sleep
        with (
            patch.object(widget, "_get_elapsed_time", return_value=elapsed_time),
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch.object(widget, "_refresh_display"),
        ):
            # Run the actual method
            await widget._update_periodically()

            # Check if interval was updated correctly
            assert widget._update_interval == expected_interval

    # Test with different elapsed times
    await test_interval_with_elapsed_time(5.0, 0.1)  # < 10s: no change
    await test_interval_with_elapsed_time(50.0, 1.0)  # 10-100s: 1.0s
    await test_interval_with_elapsed_time(500.0, 10.0)  # 100-1000s: 10.0s
    await test_interval_with_elapsed_time(1500.0, 60.0)  # >1000s: 60.0s


@pytest.mark.asyncio
async def test_maybe_async_task_status_widget_no_ipynb():
    """Test maybe_async_task_status_widget when not in ipynb."""

    async def test_task():
        return "done"

    task = asyncio.create_task(test_task())
    widget = maybe_async_task_status_widget(task)
    assert widget is None


@pytest.mark.asyncio
async def test_maybe_async_task_status_widget_with_ipywidgets():
    """Test maybe_async_task_status_widget when all requirements are met."""
    with (
        patch("pipefunc._widgets.helpers.is_running_in_ipynb", return_value=True),
    ):

        async def test_task():
            return "done"

        task = asyncio.create_task(test_task())
        widget = maybe_async_task_status_widget(task)
        assert widget is not None
        assert isinstance(widget, AsyncTaskStatusWidget)


@pytest.mark.asyncio
async def test_start_periodic_updates_error_handling():
    """Test error handling in _start_periodic_updates."""
    widget = AsyncTaskStatusWidget(display=False)

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


@pytest.mark.asyncio
async def test_update_periodically_cancellation():
    """Test that CancelledError is properly handled in _update_periodically."""
    widget = AsyncTaskStatusWidget(display=False)

    # Create a long-running mock task
    mock_task = MagicMock()
    mock_task.done.return_value = False
    widget._task = mock_task

    # Create a real task for the update loop
    update_task = asyncio.create_task(widget._update_periodically())

    # Give it a moment to start running
    await asyncio.sleep(0.1)

    # Now cancel it
    update_task.cancel()

    # This should complete without error if CancelledError is properly handled
    r = await update_task
    assert r is None
    # If CancelledError isn't handled,
    # the test will fail with an unhandled exception


@pytest.mark.asyncio
async def test_widget_error_display_without_rich():
    """Test error display when rich is not installed."""
    widget = AsyncTaskStatusWidget(display=False)
    error = ValueError("Test error message")

    # Patch has_rich to be False to simulate rich not being installed
    with (
        patch("pipefunc._widgets.async_status_widget.has_rich", new=False),
        patch("builtins.print") as mock_print,
    ):
        widget._refresh_display("failed", error)
        widget._toggle_traceback(None)
        mock_print.assert_called_with(error)
