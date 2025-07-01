# pipefunc/_widgets/progress_base.py
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pipefunc._utils import clip

if TYPE_CHECKING:
    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc.map._progress import Status


class ProgressTrackerBase(ABC):
    """Base class for progress trackers with auto-update functionality."""

    MIN_AUTO_UPDATE_INTERVAL = 0.1
    MAX_AUTO_UPDATE_INTERVAL = 10.0
    FIRST_AUTO_UPDATE_INTERVAL = 1.0
    SYNC_UPDATE_INTERVAL = 0.01
    INITIAL_UPDATE_PERIOD = 30.0
    INITIAL_MAX_UPDATE_INTERVAL = 1.0

    def __init__(
        self,
        progress_dict: dict[OUTPUT_TYPE, Status],
        task: asyncio.Task[Any] | None = None,
        *,
        target_progress_change: float = 0.05,
        auto_update: bool = True,
        in_async: bool = True,
    ) -> None:
        self.task: asyncio.Task[None] | None = None
        self.progress_dict: dict[OUTPUT_TYPE, Status] = progress_dict
        self.target_progress_change: float = target_progress_change
        self.auto_update: bool = auto_update
        self.in_async: bool = in_async
        self.last_update_time: float = 0.0
        self.start_time: float = 0.0
        self._auto_update_task: asyncio.Task | None = None
        self._min_auto_update_interval: float = self.MIN_AUTO_UPDATE_INTERVAL
        self._max_auto_update_interval: float = self.MAX_AUTO_UPDATE_INTERVAL
        self._first_auto_update_interval: float = self.FIRST_AUTO_UPDATE_INTERVAL
        self._sync_update_interval: float = self.SYNC_UPDATE_INTERVAL
        self._initial_update_period: float = self.INITIAL_UPDATE_PERIOD
        self._initial_max_update_interval: float = self.INITIAL_MAX_UPDATE_INTERVAL
        self._marked_completed: set[OUTPUT_TYPE] = set()
        self._completed = False
        if task is not None:
            self.attach_task(task)

    def attach_task(self, task: asyncio.Task[Any]) -> None:
        """Attach a new task to the progress tracker."""
        self.task = task
        self._set_auto_update(self.auto_update)

    @abstractmethod
    def update_progress(self, _: Any = None, *, force: bool = False) -> None:
        """Update the progress values and labels."""

    @abstractmethod
    def _mark_completed(self) -> None:
        """Mark the progress as completed."""

    @abstractmethod
    def _cancel_calculation(self, _: Any) -> None:
        """Cancel the ongoing calculation."""

    @abstractmethod
    def display(self) -> None:
        """Display the progress."""

    @abstractmethod
    def _update_auto_update_interval_text(self, new_interval: float) -> None:
        """Update the auto-update interval."""

    def _calculate_adaptive_interval_with_previous(self) -> float:
        """Calculate a dynamic interval based on progress changes for all resources."""
        min_interval = self._min_auto_update_interval
        max_interval = self._max_auto_update_interval
        shortest_interval = max_interval
        current_time = time.monotonic()
        for status in self.progress_dict.values():
            if status.progress <= 0 or status.progress >= 1:
                continue
            assert status.start_time is not None
            elapsed_time = current_time - status.start_time
            progress_rate = status.progress / elapsed_time
            estimated_time_for_target = self.target_progress_change / progress_rate
            # Estimate time for target progress change
            shortest_interval = min(shortest_interval, estimated_time_for_target)
        return min(max(shortest_interval, min_interval), max_interval)

    async def _auto_update_progress(self) -> None:
        """Periodically update the progress."""
        self.start_time = time.monotonic()
        while self.auto_update:
            self.update_progress()
            current_time = time.monotonic()
            elapsed_since_start = current_time - self.start_time

            new_interval = self._calculate_adaptive_interval_with_previous()
            if elapsed_since_start <= self._initial_update_period:
                new_interval = min(new_interval, self._initial_max_update_interval)

            if self._all_completed():
                break
            self._update_auto_update_interval_text(new_interval)
            await asyncio.sleep(new_interval)

    def _all_completed(self) -> bool:
        return all(status.progress >= 1.0 for status in self.progress_dict.values())

    def _toggle_auto_update(self, _: Any = None) -> None:
        """Toggle the auto-update feature on or off."""
        self._set_auto_update(not self.auto_update)

    def _set_auto_update(self, value: bool) -> None:
        """Set the auto-update feature to the given value."""
        self.auto_update = value
        if self.auto_update:
            self._auto_update_task = asyncio.create_task(self._auto_update_progress())
        elif self._auto_update_task is not None:
            self._auto_update_task.cancel()
            self._auto_update_task = None

    def _should_throttle_update(self, force: bool) -> bool:
        """Check if update should be throttled (only for sync mode)."""
        if self.in_async or force:
            return False
        now = time.monotonic()
        return now - self.last_update_time < self._sync_update_interval

    def _update_sync_interval(self, update_duration: float) -> None:
        """Update the sync interval based on how long the update took."""
        self._sync_update_interval = clip(50 * update_duration, 0.01, 1.0)
