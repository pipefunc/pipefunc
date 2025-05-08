"""Tests for async status widget functionality."""

import asyncio
import time
from unittest.mock import patch

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
    assert widget._widget is not None
    assert widget._start_time <= time.time()
    assert widget._update_interval == 0.1
    assert widget._task is None
    assert widget._update_timer is None


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
    await asyncio.sleep(0.1)
    elapsed = widget._get_elapsed_time()
    assert elapsed >= 0.1


@pytest.mark.asyncio
async def test_widget_html_content():
    """Test HTML content generation."""
    widget = AsyncMapStatusWidget(display=False)
    html = widget._create_html_content("running")
    assert isinstance(html, str)
    assert "⚙️" in html
    assert "Task is running" in html


@pytest.mark.asyncio
async def test_widget_error_formatting():
    """Test error message formatting."""
    widget = AsyncMapStatusWidget(display=False)
    msg = "Test error message"
    error = ValueError(msg)
    formatted = widget._format_error_message(error)
    assert "ValueError" in formatted
    assert msg in formatted


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

    task = asyncio.create_task(test_task())
    widget.attach_task(task)

    await task
    await asyncio.sleep(0.1)  # Give time for callback to execute


@pytest.mark.asyncio
async def test_widget_task_cancellation():
    """Test widget updates on task cancellation."""
    widget = AsyncMapStatusWidget(display=False)

    async def test_task():
        await asyncio.sleep(1)

    task = asyncio.create_task(test_task())
    widget.attach_task(task)

    task.cancel()
    await asyncio.sleep(0.1)  # Give time for callback to execute


@pytest.mark.asyncio
async def test_widget_task_failure():
    """Test widget updates on task failure."""
    widget = AsyncMapStatusWidget(display=False)

    msg = "Test error"

    async def test_task():
        raise ValueError(msg)

    task = asyncio.create_task(test_task())
    widget.attach_task(task)

    with pytest.raises(ValueError, match=msg):
        await task
    await asyncio.sleep(0.1)  # Give time for callback to execute


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
