from concurrent.futures import Future
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from ipywidgets import HTML, Button, FloatProgress, VBox

from pipefunc._pipeline._types import OUTPUT_TYPE
from pipefunc._widgets.progress import ProgressTracker
from pipefunc.map._progress import Status


@pytest.fixture
def mock_status() -> Status:
    return Status(
        n_total=100,
        n_in_progress=50,
        n_completed=50,
        n_failed=0,
        start_time=100,
        end_time=None,
    )


@pytest.fixture
def mock_progress_dict(mock_status: Status) -> dict[OUTPUT_TYPE, Status]:
    return {"test": mock_status}


@pytest.fixture
def mock_task() -> Mock:
    return Mock()


@pytest.mark.asyncio
async def test_progress_tracker_init(mock_progress_dict: dict[OUTPUT_TYPE, Status]) -> None:
    with patch("pipefunc._widgets.progress.IPython.display.display"):
        progress = ProgressTracker(mock_progress_dict, None)

    assert progress.task is None
    assert progress.progress_dict == mock_progress_dict
    assert isinstance(progress.progress_bars["test"], FloatProgress)
    assert isinstance(progress.labels["test"]["percentage"], HTML)
    assert isinstance(progress.buttons["update"], Button)


@pytest.mark.asyncio
async def test_progress_tracker_attach_task(mock_progress_dict):
    progress = ProgressTracker(mock_progress_dict, display=False)
    new_task = AsyncMock()
    progress.attach_task(new_task)
    assert progress.task == new_task


@pytest.mark.asyncio
async def test_progress_tracker_update_progress(
    mock_progress_dict: dict[OUTPUT_TYPE, Status],
) -> None:
    progress = ProgressTracker(mock_progress_dict, display=False)
    progress.update_progress()
    assert progress.progress_bars["test"].value == 0.5


@pytest.mark.asyncio
async def test_progress_tracker_update_labels(
    mock_progress_dict: dict[OUTPUT_TYPE, Status],
) -> None:
    progress = ProgressTracker(mock_progress_dict, display=False)
    progress._update_labels("test", mock_progress_dict["test"])
    assert "50.0%" in progress.labels["test"]["percentage"].value


@pytest.mark.asyncio
async def test_progress_tracker_calculate_adaptive_interval(
    mock_progress_dict: dict[OUTPUT_TYPE, Status],
) -> None:
    progress = ProgressTracker(mock_progress_dict, display=False)
    interval = progress._calculate_adaptive_interval_with_previous()
    assert 0.1 <= interval <= 10.0


@pytest.mark.asyncio
async def test_progress_tracker_auto_update_progress(
    mock_progress_dict: dict[OUTPUT_TYPE, Status],
) -> None:
    progress = ProgressTracker(mock_progress_dict, display=False)
    progress.auto_update = True
    update_progress_mock = Mock()
    with (
        patch.object(progress, "update_progress", update_progress_mock),
        patch.object(progress, "_calculate_adaptive_interval_with_previous", return_value=0.1),
        patch.object(progress, "_all_completed", side_effect=[False, True]),
        patch("asyncio.sleep", return_value=None),  # Mock asyncio.sleep
    ):
        await progress._auto_update_progress()
    assert progress.auto_update_interval_label.value != "Auto-update every: N/A"
    assert update_progress_mock.call_count == 2  # Ensure update_progress was called twice


@pytest.mark.asyncio
async def test_progress_tracker_all_completed(
    mock_progress_dict: dict[OUTPUT_TYPE, Status],
) -> None:
    progress = ProgressTracker(mock_progress_dict, display=False)
    assert not progress._all_completed()
    mock_progress_dict["test"].mark_complete(n=50)
    assert progress._all_completed()


@pytest.mark.asyncio
async def test_progress_tracker_mark_completed(
    mock_progress_dict: dict[OUTPUT_TYPE, Status],
) -> None:
    progress = ProgressTracker(mock_progress_dict, display=False)
    progress._mark_completed()
    assert all(button.disabled for button in progress.buttons.values())
    assert "Completed all tasks" in progress.auto_update_interval_label.value


@pytest.mark.asyncio
async def test_progress_tracker_toggle_auto_update(
    mock_progress_dict: dict[OUTPUT_TYPE, Status],
) -> None:
    progress = ProgressTracker(mock_progress_dict, display=False)
    initial_state = progress.auto_update
    progress._toggle_auto_update()
    assert progress.auto_update != initial_state


@pytest.mark.asyncio
async def test_progress_tracker_set_auto_update(
    mock_progress_dict: dict[OUTPUT_TYPE, Status],
) -> None:
    progress = ProgressTracker(mock_progress_dict, display=False)
    progress._set_auto_update(value=True)
    assert progress.auto_update
    assert progress.buttons["toggle_auto_update"].description == "Stop Auto-Update"
    progress._set_auto_update(value=False)
    assert not progress.auto_update
    assert progress.buttons["toggle_auto_update"].description == "Start Auto-Update"


@pytest.mark.asyncio
async def test_progress_tracker_cancel_calculation(
    mock_progress_dict: dict[OUTPUT_TYPE, Status],
    mock_task: Mock,
) -> None:
    progress = ProgressTracker(mock_progress_dict, mock_task, display=False)
    progress._cancel_calculation(None)
    assert mock_task.cancel.called
    assert all(button.disabled for button in progress.buttons.values())
    assert "Calculation cancelled" in progress.auto_update_interval_label.value


@pytest.mark.asyncio
async def test_progress_tracker_widgets(mock_progress_dict: dict[OUTPUT_TYPE, Status]) -> None:
    progress = ProgressTracker(mock_progress_dict, display=False)
    widgets = progress._widgets
    assert isinstance(widgets, VBox)
    assert len(widgets.children) > 0


@pytest.mark.asyncio
async def test_progress_tracker_display(mock_progress_dict: dict[OUTPUT_TYPE, Status]) -> None:
    with patch("pipefunc._widgets.progress.IPython.display.display") as mock_display:
        progress = ProgressTracker(mock_progress_dict, display=False)
        progress.display()
        assert mock_display.call_count == 2  # display on HTML and VBox


@pytest.mark.asyncio
async def test_progress_tracker_failed(mock_progress_dict: dict[OUTPUT_TYPE, Status]) -> None:
    progress = ProgressTracker(mock_progress_dict, display=False)
    assert not progress._all_completed()
    mock_progress_dict["test"].mark_complete(n=49)
    future: Future[Any] = Future()
    future.set_exception(Exception("Failed"))
    mock_progress_dict["test"].mark_complete(n=1, future=future)
    assert progress._all_completed()


@pytest.mark.asyncio
async def test_progress_tracker_color_by_scope():
    progress_dict = {
        "foo.a": Status(n_total=100),
        "foo.b": Status(n_total=100),
        "bar.a": Status(n_total=100),
        "bar.b": Status(n_total=100),
        "c": Status(n_total=100),
    }
    mock_task = AsyncMock()
    progress = ProgressTracker(progress_dict, mock_task, display=False)

    widgets = progress._widgets
    assert len(widgets.children) == len(progress_dict) + 2
    borders = {child.layout.border for child in widgets.children[: len(progress_dict)]}
    assert len(borders) == 3  # 3 different scopes, foo, bar, None
    with patch("pipefunc._widgets.progress.IPython.display.display") as mock_display:
        progress.display()
        assert mock_display.call_count == 2
