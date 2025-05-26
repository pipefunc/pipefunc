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

if TYPE_CHECKING:
    import asyncio

    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc.map._progress import Status


class RichProgressTracker:
    """Text-based progress tracker using rich.progress with the same interface as ProgressTracker."""

    def __init__(
        self,
        progress_dict: dict[OUTPUT_TYPE, Status],
        task: asyncio.Task[Any] | None = None,
        *,
        display: bool = True,
        in_async: bool = True,
        refresh_interval: float = 0.1,
    ) -> None:
        self.task: asyncio.Task[None] | None = task
        self.progress_dict: dict[OUTPUT_TYPE, Status] = progress_dict
        self._marked_completed: set[OUTPUT_TYPE] = set()
        self.in_async: bool = in_async
        self.last_update_time: float = 0.0
        self.refresh_interval: float = refresh_interval

        # Rich-specific attributes
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[status]}"),
            console=self.console,
            auto_refresh=False,
        )
        self.task_ids: dict[OUTPUT_TYPE, TaskID] = {}

        # Create tasks in the progress bar
        for name, status in self.progress_dict.items():
            description = ", ".join(at_least_tuple(name))
            task_id = self.progress.add_task(
                description,
                total=status.n_total,
                completed=status.n_completed,
                status=self._get_status_text(status),
            )
            self.task_ids[name] = task_id

        if display:
            self.display()

    def _get_status_text(self, status: Status) -> str:
        """Generate status text for a task."""
        if status.n_failed == 0:
            return f"✅ {status.n_completed:,} | ⏳ {status.n_left:,}"
        return f"✅ {status.n_completed:,} | ❌ {status.n_failed:,} | ⏳ {status.n_left:,}"

    def attach_task(self, task: asyncio.Task[Any]) -> None:
        """Attach a new task to the progress tracker."""
        self.task = task

    def update_progress(self, _: Any = None, *, force: bool = False) -> None:  # noqa: ARG002
        """Update the progress values."""
        for name, status in self.progress_dict.items():
            if status.progress == 0 or name in self._marked_completed:
                continue

            task_id = self.task_ids[name]
            self.progress.update(
                task_id,
                completed=status.n_completed,
                status=self._get_status_text(status),
            )

            if status.progress >= 1.0:
                self._marked_completed.add(name)
                txt = ", ".join(at_least_tuple(name))
                if status.n_failed == 0:
                    description = f"[bold green]✅ {txt}[/bold green]"
                else:
                    description = f"[bold red]❌ {txt}[/bold red]"
                self.progress.update(task_id, description=description)

        if self._all_completed():
            self._mark_completed()

        self._maybe_refresh()

    def _maybe_refresh(self) -> None:
        now = time.monotonic()
        if now - self.last_update_time > self.refresh_interval:
            self.progress.refresh()
            self.last_update_time = now

    def _all_completed(self) -> bool:
        return all(status.progress >= 1.0 for status in self.progress_dict.values())

    def _mark_completed(self) -> None:
        if any(status.n_failed > 0 for status in self.progress_dict.values()):
            self.console.print("\n[bold red]Completed with errors ❌[/bold red]")
        else:
            self.console.print("\n[bold green]Completed all tasks 🎉[/bold green]")
        self.stop()

    def _cancel_calculation(self, _: Any) -> None:
        """Cancel the ongoing calculation."""
        if self.task is not None:
            self.task.cancel()
        self.update_progress()  # Update progress one last time
        self.console.print("\n[bold red]Calculation cancelled ❌[/bold red]")

        # Mark incomplete tasks as cancelled
        for name, status in self.progress_dict.items():
            if status.progress < 1.0:
                task_id = self.task_ids[name]
                description = (
                    f"[bold red]❌ {', '.join(at_least_tuple(name))} (cancelled)[/bold red]"
                )
                self.progress.update(task_id, description=description)

        self.stop()

    def display(self) -> None:
        """Display the progress bars using Rich Live display."""
        # self.progress.start()

    def stop(self) -> None:
        """Stop the live display."""
        self.progress.refresh()
        self.progress.stop()

    @property
    def auto_update(self) -> bool:
        """Always True as Rich handles updates automatically."""
        return True
