from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from ipywidgets import HTML, Button, FloatProgress, VBox

from pipefunc._widgets import (
    ProgressTracker,
)


@pytest.fixture
def mock_status():
    return MagicMock(
        progress=0.5,
        n_completed=50,
        n_left=50,
        start_time=100,
        end_time=None,
        elapsed_time=MagicMock(return_value=10),
    )


@pytest.fixture
def mock_progress_dict(mock_status):
    return {"test": mock_status}


@pytest.fixture
def mock_task():
    return AsyncMock()


@pytest.mark.asyncio
async def test_progress_tracker_init(mock_progress_dict, mock_task):
    with patch("pipefunc._widgets.IPython.display.display"):
        tracker = ProgressTracker(mock_progress_dict, mock_task)

    assert tracker.task == mock_task
    assert tracker.progress_dict == mock_progress_dict
    assert isinstance(tracker.progress_bars["test"], FloatProgress)
    assert isinstance(tracker.labels["test"]["percentage"], HTML)
    assert isinstance(tracker.buttons["update"], Button)


@pytest.mark.asyncio
async def test_progress_tracker_attach_task(mock_progress_dict):
    tracker = ProgressTracker(mock_progress_dict, display=False)
    new_task = AsyncMock()
    tracker.attach_task(new_task)
    assert tracker.task == new_task


@pytest.mark.asyncio
async def test_progress_tracker_update_progress(mock_progress_dict, mock_task):
    tracker = ProgressTracker(mock_progress_dict, mock_task, display=False)
    tracker.update_progress()
    assert tracker.progress_bars["test"].value == 0.5


@pytest.mark.asyncio
async def test_progress_tracker_update_labels(mock_progress_dict, mock_task):
    tracker = ProgressTracker(mock_progress_dict, mock_task, display=False)
    tracker._update_labels("test", mock_progress_dict["test"])
    assert "50.0%" in tracker.labels["test"]["percentage"].value


@pytest.mark.asyncio
async def test_progress_tracker_calculate_adaptive_interval(mock_progress_dict, mock_task):
    tracker = ProgressTracker(mock_progress_dict, mock_task, display=False)
    interval = tracker._calculate_adaptive_interval_with_previous()
    assert 0.1 <= interval <= 10.0


@pytest.mark.asyncio
async def test_progress_tracker_auto_update_progress(mock_progress_dict, mock_task):
    tracker = ProgressTracker(mock_progress_dict, mock_task, display=False)
    tracker.auto_update = True
    tracker.first_update = False
    update_progress_mock = Mock()
    with (
        patch.object(tracker, "update_progress", update_progress_mock),
        patch.object(tracker, "_calculate_adaptive_interval_with_previous", return_value=0.1),
        patch.object(tracker, "_all_completed", side_effect=[False, True]),
        patch("asyncio.sleep", return_value=None),  # Mock asyncio.sleep
    ):
        await tracker._auto_update_progress()
    assert tracker.auto_update_interval_label.value != "Auto-update every: N/A"
    assert update_progress_mock.call_count == 2  # Ensure update_progress was called twice


@pytest.mark.asyncio
async def test_progress_tracker_all_completed(mock_progress_dict, mock_task):
    tracker = ProgressTracker(mock_progress_dict, mock_task, display=False)
    assert not tracker._all_completed()
    mock_progress_dict["test"].progress = 1.0
    assert tracker._all_completed()


@pytest.mark.asyncio
async def test_progress_tracker_mark_completed(mock_progress_dict, mock_task):
    tracker = ProgressTracker(mock_progress_dict, mock_task, display=False)
    tracker._mark_completed()
    assert all(button.disabled for button in tracker.buttons.values())
    assert "Completed all tasks" in tracker.auto_update_interval_label.value


@pytest.mark.asyncio
async def test_progress_tracker_toggle_auto_update(mock_progress_dict, mock_task):
    tracker = ProgressTracker(mock_progress_dict, mock_task, display=False)
    initial_state = tracker.auto_update
    tracker._toggle_auto_update()
    assert tracker.auto_update != initial_state


@pytest.mark.asyncio
async def test_progress_tracker_set_auto_update(mock_progress_dict, mock_task):
    tracker = ProgressTracker(mock_progress_dict, mock_task, display=False)
    tracker._set_auto_update(value=True)
    assert tracker.auto_update
    assert tracker.buttons["toggle_auto_update"].description == "Stop Auto-Update"
    tracker._set_auto_update(value=False)
    assert not tracker.auto_update
    assert tracker.buttons["toggle_auto_update"].description == "Start Auto-Update"


@pytest.mark.asyncio
async def test_progress_tracker_cancel_calculation(mock_progress_dict, mock_task):
    tracker = ProgressTracker(mock_progress_dict, mock_task, display=False)
    tracker._cancel_calculation(None)
    assert mock_task.cancel.called
    assert all(button.disabled for button in tracker.buttons.values())
    assert "Calculation cancelled" in tracker.auto_update_interval_label.value


@pytest.mark.asyncio
async def test_progress_tracker_widgets(mock_progress_dict, mock_task):
    tracker = ProgressTracker(mock_progress_dict, mock_task, display=False)
    widgets = tracker._widgets()
    assert isinstance(widgets, VBox)
    assert len(widgets.children) > 0


@pytest.mark.asyncio
async def test_progress_tracker_display(mock_progress_dict, mock_task):
    with patch("pipefunc._widgets.IPython.display.display") as mock_display:
        tracker = ProgressTracker(mock_progress_dict, mock_task, display=False)
        tracker.display()
        assert mock_display.call_count == 2  # display on HTML and VBox
