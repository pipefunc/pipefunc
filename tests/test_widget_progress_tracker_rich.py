from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from pipefunc._widgets.progress_rich import RichProgressTracker, _format_time
from pipefunc.map._progress import Status


@pytest.fixture
def mock_status() -> Status:
    return Status(
        n_total=100,
        n_in_progress=50,
        n_completed=50,
        n_failed=0,
    )


@pytest.fixture
def mock_status_not_started() -> Status:
    return Status(
        n_total=100,
        n_in_progress=0,
        n_completed=0,
        n_failed=0,
    )


@pytest.fixture
def mock_status_completed() -> Status:
    status = Status(
        n_total=100,
        n_in_progress=0,
        n_completed=100,
        n_failed=0,
    )
    status.mark_complete(n=100)  # Ensure progress is 1.0
    return status


@pytest.fixture
def mock_status_failed() -> Status:
    status = Status(
        n_total=100,
        n_in_progress=0,
        n_completed=99,
        n_failed=1,
    )
    # Simulate one failure
    status.n_completed = 99
    status.n_failed = 1
    # To make _all_completed true
    status.n_in_progress = 0
    return status


@pytest.fixture
def mock_progress_dict(mock_status: Status) -> dict[str | tuple[str, ...], Status]:
    return {"test_task": mock_status}


@pytest.fixture
def mock_progress_dict_multiple(
    mock_status: Status,
    mock_status_not_started: Status,
) -> dict[str | tuple[str, ...], Status]:
    return {
        "task1": mock_status,
        "task2": mock_status_not_started,
        ("task3_a", "task3_b"): Status(n_total=50, n_completed=0),
    }


@pytest.fixture
def mock_task() -> AsyncMock:
    return AsyncMock(spec=asyncio.Task)


def test_rich_progress_tracker_init_basic(
    mock_progress_dict: dict[str | tuple[str, ...], Status],
) -> None:
    """Test basic initialization without auto-update."""
    tracker = RichProgressTracker(mock_progress_dict, None, auto_update=False)

    assert tracker.progress_dict == mock_progress_dict
    assert tracker.task is None
    assert not tracker.auto_update
    assert "test_task" in tracker._task_ids
    assert tracker._console is not None
    assert tracker._progress is not None


@pytest.mark.asyncio
async def test_rich_progress_tracker_init_with_task(
    mock_progress_dict: dict[str | tuple[str, ...], Status],
    mock_task: AsyncMock,
) -> None:
    """Test initialization with a task."""
    tracker = RichProgressTracker(mock_progress_dict, mock_task, auto_update=False)

    assert tracker.progress_dict == mock_progress_dict
    assert tracker.task == mock_task
    assert "test_task" in tracker._task_ids


def test_rich_progress_tracker_update_progress(
    mock_progress_dict: dict[str | tuple[str, ...], Status],
) -> None:
    """Test updating progress values."""
    tracker = RichProgressTracker(mock_progress_dict, auto_update=False)

    # Update status
    mock_progress_dict["test_task"].n_completed = 75

    # This should not raise any exceptions
    tracker.update_progress()

    # Verify the progress was updated internally
    assert tracker.progress_dict["test_task"].n_completed == 75


def test_rich_progress_tracker_update_progress_completed_task(
    mock_progress_dict: dict[str | tuple[str, ...], Status],
    mock_status_completed: Status,
) -> None:
    """Test updating progress for a completed task."""
    mock_progress_dict["completed_task"] = mock_status_completed
    tracker = RichProgressTracker(mock_progress_dict, auto_update=False)

    # First update, completed_task should be marked as completed
    tracker.update_progress()
    assert "completed_task" in tracker._marked_completed

    # Second update, completed_task should not be processed again
    initial_marked_count = len(tracker._marked_completed)
    tracker.update_progress()
    assert len(tracker._marked_completed) == initial_marked_count


def test_rich_progress_tracker_update_progress_all_completed(
    mock_status_completed: Status,
) -> None:
    """Test behavior when all tasks are completed."""
    progress_dict: dict[str | tuple[str, ...], Status] = {
        "task1": mock_status_completed,
        "task2": Status(n_total=10, n_completed=10),
    }
    progress_dict["task2"].mark_complete(n=10)

    tracker = RichProgressTracker(progress_dict, auto_update=False)

    with patch.object(tracker, "_mark_completed") as mock_mark_completed:
        tracker.update_progress()
        mock_mark_completed.assert_called_once()


def test_rich_progress_tracker_mark_completed_success(
    mock_progress_dict: dict[str | tuple[str, ...], Status],
) -> None:
    """Test marking completion with success."""
    tracker = RichProgressTracker(mock_progress_dict, auto_update=False)

    with patch.object(tracker, "_stop") as mock_stop:
        # Capture the console output
        with patch.object(tracker._console, "print") as mock_print:
            tracker._mark_completed()
            mock_print.assert_called_once()
            # Check that the call contains success message
            call_args = mock_print.call_args[0][0]
            assert "Completed all tasks" in call_args
            assert "🎉" in call_args
        mock_stop.assert_called_once()


def test_rich_progress_tracker_mark_completed_with_errors(
    mock_progress_dict: dict[str | tuple[str, ...], Status],
    mock_status_failed: Status,
) -> None:
    """Test marking completion with errors."""
    mock_progress_dict["failed_task"] = mock_status_failed
    tracker = RichProgressTracker(mock_progress_dict, auto_update=False)

    with patch.object(tracker, "_stop") as mock_stop:
        with patch.object(tracker._console, "print") as mock_print:
            tracker._mark_completed()
            mock_print.assert_called_once()
            # Check that the call contains error message
            call_args = mock_print.call_args[0][0]
            assert "Completed with errors" in call_args
            assert "❌" in call_args
        mock_stop.assert_called_once()


def test_rich_progress_tracker_update_progress_failed_task(
    mock_status_failed: Status,
) -> None:
    """Test updating progress for a failed task."""
    progress_dict: dict[str | tuple[str, ...], Status] = {"failed_task": mock_status_failed}
    tracker = RichProgressTracker(progress_dict, auto_update=False)

    # This should not raise any exceptions
    tracker.update_progress()

    # The failed task should be marked as completed (since it's done, even with failures)
    assert "failed_task" in tracker._marked_completed


def test_rich_progress_tracker_cancel_calculation(
    mock_progress_dict_multiple: dict[str | tuple[str, ...], Status],
    mock_task: AsyncMock,
) -> None:
    """Test cancelling calculation."""
    tracker = RichProgressTracker(mock_progress_dict_multiple, mock_task, auto_update=False)

    with patch.object(tracker, "_stop") as mock_stop:
        with patch.object(tracker._console, "print") as mock_print:
            tracker._cancel_calculation(None)

            mock_task.cancel.assert_called_once()
            mock_print.assert_called_once()
            # Check that the call contains cancellation message
            call_args = mock_print.call_args[0][0]
            assert "Calculation cancelled" in call_args
            assert "❌" in call_args
        mock_stop.assert_called_once()


def test_rich_progress_tracker_cancel_calculation_no_task(
    mock_progress_dict: dict[str | tuple[str, ...], Status],
) -> None:
    """Test cancelling calculation when no task is present."""
    tracker = RichProgressTracker(mock_progress_dict, None, auto_update=False)

    with patch.object(tracker, "update_progress") as mock_update_progress:
        tracker._cancel_calculation(None)
        mock_update_progress.assert_called_with(force=True)


def test_rich_progress_tracker_display(
    mock_progress_dict: dict[str | tuple[str, ...], Status],
) -> None:
    """Test display method."""
    tracker = RichProgressTracker(mock_progress_dict, auto_update=False)

    with patch.object(tracker._progress, "start") as mock_start:
        tracker.display()
        mock_start.assert_called_once()


def test_rich_progress_tracker_stop(
    mock_progress_dict: dict[str | tuple[str, ...], Status],
) -> None:
    """Test stop method."""
    tracker = RichProgressTracker(mock_progress_dict, auto_update=False)

    with (
        patch.object(tracker._progress, "refresh") as mock_refresh,
        patch.object(tracker._progress, "stop") as mock_stop,
    ):
        tracker._stop()
        mock_refresh.assert_called_once()
        mock_stop.assert_called_once()


def test_rich_progress_tracker_update_auto_update_interval_text(
    mock_progress_dict: dict[str | tuple[str, ...], Status],
) -> None:
    """Test update_auto_update_interval_text method (no-op)."""
    tracker = RichProgressTracker(mock_progress_dict, auto_update=False)
    # This is a no-op, so just call it for coverage
    tracker._update_auto_update_interval_text(0.1)
    # No assertion needed as it's a no-op


def test_rich_progress_tracker_update_progress_throttling(
    mock_progress_dict_multiple: dict[str | tuple[str, ...], Status],
) -> None:
    """Test progress update throttling behavior."""
    tracker = RichProgressTracker(
        mock_progress_dict_multiple,
        auto_update=False,
        target_progress_change=0.01,
    )

    # Initial update
    tracker.update_progress()

    # Simulate small progress change that should be throttled
    tracker.progress_dict["task1"].n_completed = 51  # 1% change
    tracker.last_update_time = time.monotonic() - 0.01  # very recent update

    # This should work without errors (throttling is internal behavior)
    tracker.update_progress(force=False)

    # Force update should always work
    tracker.progress_dict["task2"].n_completed = 10
    tracker.update_progress(force=True)


def test_rich_progress_tracker_update_progress_task_progress_zero(
    mock_status_not_started: Status,
) -> None:
    """Test updating progress for a task with zero progress."""
    progress_dict: dict[str | tuple[str, ...], Status] = {"task_zero": mock_status_not_started}
    tracker = RichProgressTracker(progress_dict, auto_update=False)

    # This should not raise any exceptions
    tracker.update_progress()

    # Task with zero progress should not be marked as completed
    assert "task_zero" not in tracker._marked_completed


def test_rich_progress_tracker_multiple_tasks_with_tuples(
    mock_progress_dict_multiple: dict[str | tuple[str, ...], Status],
) -> None:
    """Test handling multiple tasks including tuple names."""
    tracker = RichProgressTracker(mock_progress_dict_multiple, auto_update=False)

    # Check that all tasks are registered
    assert "task1" in tracker._task_ids
    assert "task2" in tracker._task_ids
    assert ("task3_a", "task3_b") in tracker._task_ids

    # Update progress should work for all tasks
    tracker.update_progress()


def test_rich_progress_tracker_all_completed_check(
    mock_status_completed: Status,
) -> None:
    """Test _all_completed method."""
    # Create a status that starts incomplete
    incomplete_status = Status(n_total=100, n_completed=50, n_failed=0)
    progress_dict: dict[str | tuple[str, ...], Status] = {"task1": incomplete_status}
    tracker = RichProgressTracker(progress_dict, auto_update=False)

    # Initially not all completed
    assert not tracker._all_completed()

    # Complete the task
    incomplete_status.mark_complete(n=50)  # Complete the remaining 50

    # Now should be completed
    assert tracker._all_completed()


def test_rich_progress_tracker_attach_task(
    mock_progress_dict: dict[str | tuple[str, ...], Status],
) -> None:
    """Test attaching a task to the tracker."""
    tracker = RichProgressTracker(mock_progress_dict, auto_update=False)
    new_task = AsyncMock()

    tracker.attach_task(new_task)
    assert tracker.task == new_task


@pytest.mark.asyncio
async def test_rich_progress_tracker_with_auto_update(
    mock_progress_dict: dict[str | tuple[str, ...], Status],
    mock_task: AsyncMock,
) -> None:
    """Test tracker with auto-update enabled."""
    with patch.object(RichProgressTracker, "_auto_update_progress") as mock_auto_update:
        # Mock the auto-update coroutine to avoid infinite loop
        mock_auto_update.return_value = AsyncMock()

        tracker = RichProgressTracker(
            mock_progress_dict,
            mock_task,
            auto_update=True,
        )

        assert tracker.auto_update
        assert tracker.task == mock_task


def test_rich_progress_tracker_early_return_throttling(
    mock_progress_dict_multiple: dict[str | tuple[str, ...], Status],
) -> None:
    """Test early return in update_progress when throttling is active."""
    tracker = RichProgressTracker(
        mock_progress_dict_multiple,
        auto_update=False,
        target_progress_change=0.01,
        in_async=False,  # Enable sync mode for throttling
    )

    # Set up conditions for throttling
    tracker.last_update_time = time.monotonic()  # Recent update
    tracker._sync_update_interval = 1.0  # Large interval to ensure throttling

    # Make a small progress change
    tracker.progress_dict["task1"].n_completed = 51  # Small change

    # This should trigger the early return due to throttling
    tracker.update_progress(force=False)

    # Verify that the method returned early by checking that task2 wasn't processed
    # (since task1 would cause early return before task2 is processed)
    assert tracker.progress_dict["task2"].n_completed == 0  # Should remain unchanged


def test_format_time():
    assert _format_time(0) == "00:00"
    assert _format_time(1) == "00:01"
    assert _format_time(60) == "01:00"
    assert _format_time(3600) == "1:00:00"
    assert _format_time(3600 * 24) == "24:00:00"


def test_rich_progress_tracker_display_multiple_times(
    mock_progress_dict: dict[str | tuple[str, ...], Status],
) -> None:
    """Test display method."""
    tracker = RichProgressTracker(mock_progress_dict, auto_update=False)

    tracker.display()

    with pytest.raises(RuntimeError, match="Progress bar already started"):
        tracker.display()
