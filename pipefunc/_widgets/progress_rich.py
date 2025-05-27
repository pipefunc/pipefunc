# pipefunc/_widgets/progress_rich.py
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pipefunc._utils import at_least_tuple
from pipefunc._widgets.progress_base import ProgressTrackerBase

if TYPE_CHECKING:
    import asyncio

    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc.map._progress import Status


class RichProgressTracker(ProgressTrackerBase):
    """Text-based progress tracker using rich.progress."""

    def __init__(
        self,
        progress_dict: dict[OUTPUT_TYPE, Status],
        task: asyncio.Task[Any] | None = None,
        *,
        target_progress_change: float = 0.05,
        auto_update: bool = True,
        display: bool = True,
        in_async: bool = True,
    ) -> None:
        super().__init__(
            progress_dict,
            task,
            target_progress_change=target_progress_change,
            auto_update=auto_update,
            in_async=in_async,
        )

        # Rich-specific attributes
        self._console = Console()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[status]}"),
            console=self._console,
            auto_refresh=False,
        )
        self._task_ids: dict[OUTPUT_TYPE, TaskID] = {}

        # Create tasks in the progress bar
        for name, status in self.progress_dict.items():
            description = ", ".join(at_least_tuple(name))
            task_id = self._progress.add_task(
                description,
                total=status.n_total,
                completed=status.n_completed,
                status=self.get_status_text(status),
            )
            self._task_ids[name] = task_id

        if display:
            self.display()
        if self.task is not None:
            self._set_auto_update(auto_update)

    def update_progress(self, _: Any = None, *, force: bool = False) -> None:
        """Update the progress values."""
        now = time.monotonic()

        if self._should_throttle_update(force):
            return

        for name, status in self.progress_dict.items():
            if status.progress == 0 or name in self._marked_completed:
                continue

            task_id = self._task_ids[name]
            self._progress.update(
                task_id,
                total=status.n_total,
                completed=status.n_completed,
                status=self.get_status_text(status),
            )

            if status.progress >= 1.0:
                self._marked_completed.add(name)
                txt = ", ".join(at_least_tuple(name))
                if status.n_failed == 0:
                    description = f"[bold green]✅ {txt}[/bold green]"
                else:
                    description = f"[bold red]❌ {txt}[/bold red]"
                self._progress.update(task_id, description=description)

        if self._all_completed():
            self._mark_completed()

        self._progress.refresh()
        self.last_update_time = time.monotonic()
        self._update_sync_interval(self.last_update_time - now)

    def _mark_completed(self) -> None:
        if any(status.n_failed > 0 for status in self.progress_dict.values()):
            self._console.print("\n[bold red]Completed with errors ❌[/bold red]")
        else:
            self._console.print("\n[bold green]Completed all tasks 🎉[/bold green]")
        self._stop()

    def _cancel_calculation(self, _: Any) -> None:
        """Cancel the ongoing calculation."""
        if self.task is not None:
            self.task.cancel()
        self.update_progress(force=True)
        self._console.print("\n[bold red]Calculation cancelled ❌[/bold red]")

        # Mark incomplete tasks as cancelled
        for name, status in self.progress_dict.items():
            if status.progress < 1.0:
                task_id = self._task_ids[name]
                description = (
                    f"[bold red]❌ {', '.join(at_least_tuple(name))} (cancelled)[/bold red]"
                )
                self._progress.update(task_id, description=description)

        self._stop()

    def _update_auto_update_interval_text(self, new_interval: float) -> None:
        """Update the auto-update interval text."""
        # no-op

    def display(self) -> None:
        """Display the progress bars using Rich Live display."""
        self._progress.start()

    def _stop(self) -> None:
        """Stop the live display."""
        self._progress.refresh()
        self._progress.stop()
