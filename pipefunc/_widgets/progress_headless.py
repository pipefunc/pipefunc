from __future__ import annotations

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
        self._set_auto_update(value=False)

    def update_progress(self, _: Any = None, *, force: bool = False) -> None:
        """Update the progress values."""
        # no-op

    def _mark_completed(self) -> None:
        """Mark the progress as completed."""
        # no-op

    def _cancel_calculation(self, _: Any) -> None:  # pragma: no cover
        """Cancel the ongoing calculation."""
        if self.task is not None:
            self.task.cancel()

    def _update_auto_update_interval_text(self, new_interval: float) -> None:
        """Update the auto-update interval text."""
        # no-op

    def display(self) -> None:
        """Display the progress bars."""
        # no-op
