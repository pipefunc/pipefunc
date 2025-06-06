from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pipefunc._widgets.progress_base import ProgressTrackerBase

if TYPE_CHECKING:
    import asyncio

    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc.map._progress import Status


class HeadlessProgressTracker(ProgressTrackerBase):
    """Simple progress tracker without a UI.

    This is useful for running pipelines in a non-interactive environment.
    """

    def __init__(
        self,
        progress_dict: dict[OUTPUT_TYPE, Status],
        task: asyncio.Task[Any] | None = None,
        *,
        target_progress_change: float = 0.05,
        auto_update: bool = True,
        in_async: bool = True,
    ) -> None:
        super().__init__(
            progress_dict,
            task,
            target_progress_change=target_progress_change,
            auto_update=auto_update,
            in_async=in_async,
        )

        if self.task is not None:
            self._set_auto_update(auto_update)

    def update_progress(self, _: Any = None, *, force: bool = False) -> None:
        """Update the progress values."""
        t_start = time.monotonic()
        return_early = self._should_throttle_update(force)
        for name, status in self.progress_dict.items():
            if status.progress == 0 or name in self._marked_completed:
                continue
            if return_early and status.progress < 1.0:
                return
            if status.progress >= 1.0:
                self._marked_completed.add(name)

        if self._all_completed():
            self._mark_completed()

        self.last_update_time = time.monotonic()
        self._update_sync_interval(self.last_update_time - t_start)

    def _mark_completed(self) -> None:
        self._completed = True

    def _cancel_calculation(self, _: Any) -> None:
        """Cancel the ongoing calculation."""
        if self.task is not None:
            self.task.cancel()
        self.update_progress(force=True)

    def _update_auto_update_interval_text(self, new_interval: float) -> None:
        """Update the auto-update interval text."""
        # no-op

    def display(self) -> None:
        """Display the progress bars using Rich Live display."""
        # no-op
