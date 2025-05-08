from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import ipywidgets
from IPython.display import HTML, display

StatusType = Literal["initializing", "running", "done", "cancelled", "failed"]


@dataclass(frozen=True)
class StyleInfo:
    """Style information for a status type."""

    color: str
    icon: str
    message: str


_STYLES: dict[StatusType, StyleInfo] = {
    "initializing": StyleInfo(
        color="#3498db",  # Blue
        icon="⏳",
        message="Task is initializing...",
    ),
    "running": StyleInfo(
        color="#3498db",  # Blue
        icon="⚙️",
        message="Task is running...",
    ),
    "done": StyleInfo(
        color="#2ecc71",  # Green
        icon="✅",
        message="Task completed successfully",
    ),
    "cancelled": StyleInfo(
        color="#f39c12",  # Orange
        icon="⚠️",
        message="Task was cancelled",
    ),
    "failed": StyleInfo(
        color="#e74c3c",  # Red
        icon="❌",
        message="Task failed",
    ),
}


class AsyncMapStatusWidget:
    """Manages an ipywidgets.Output widget to display the status of an asyncio.Task."""

    def __init__(self, initial_update_interval: float = 0.1) -> None:
        """Initialize the status widget."""
        self._widget = ipywidgets.Output()
        self._start_time = time.time()
        self._start_datetime = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
        self._update_interval = initial_update_interval
        self._task: asyncio.Task | None = None
        self._update_timer: asyncio.Task | None = None

        # Initialize display
        self._refresh_display("initializing")

    def _get_elapsed_time(self) -> float:
        """Get elapsed time in seconds since widget creation."""
        return time.time() - self._start_time

    def _get_style(self, status: StatusType) -> StyleInfo:
        """Get style information for the given status."""
        return _STYLES.get(status, _STYLES["running"])

    def _create_html_content(self, status: StatusType, error: Exception | None = None) -> str:
        """Create the HTML content for the status display."""
        style = self._get_style(status)
        elapsed = self._get_elapsed_time()

        # Main container styles
        div_style = "; ".join(
            [
                "display: flex",
                "align-items: center",
                "font-size: 14px",
                "padding: 5px 10px",
                "border-radius: 4px",
                "background-color: #f8f9fa",
                f"border-left: 3px solid {style.color}",
            ],
        )

        # Build the HTML structure
        html = f"""
        <div style="{div_style}">
            <span style="color: {style.color}; font-weight: bold; margin-right: 5px;">
                {style.icon} {style.message}
            </span>
            <span style="color: #666; margin-left: auto; font-size: 12px;">
                Started: {self._start_datetime} | Elapsed: {elapsed:.1f}s
            </span>
        """

        # Add error details if applicable
        if status == "failed" and error is not None:
            error_text = self._format_error_message(error)
            html += f"""
            <span style="margin-left: 10px; color: #e74c3c; font-size: 12px;">
                {type(error).__name__}: {error_text}
            </span>
            """

        html += "</div>"
        return html

    def _format_error_message(self, error: Exception) -> str:
        """Format and truncate error message if needed."""
        error_text = str(error)
        max_chars = 300

        if len(error_text) > max_chars:
            return error_text[: max_chars - 3] + "..."
        return error_text

    def _refresh_display(self, status: StatusType, error: Exception | None = None) -> None:
        """Refresh the display with current status."""
        if self._widget is None:
            return

        with self._widget:
            self._widget.clear_output(wait=True)
            html_content = self._create_html_content(status, error)
            display(HTML(html_content))

    async def _update_periodically(self) -> None:
        """Periodically update the widget while the task is running."""
        if self._task is None:
            return

        try:
            while not self._task.done():
                self._refresh_display("running")
                await asyncio.sleep(self._update_interval)
                elapsed = self._get_elapsed_time()
                if elapsed > 10:  # noqa: PLR2004
                    self._update_interval = 1.0
                elif elapsed > 100:  # noqa: PLR2004
                    self._update_interval = 10.0
                elif elapsed > 1000:  # noqa: PLR2004
                    self._update_interval = 60.0
        except asyncio.CancelledError:
            # Expected when the main task finishes
            pass
        except Exception as e:  # noqa: BLE001
            print(f"Error in periodic update: {e}")

    def _stop_periodic_updates(self) -> None:
        """Stop the periodic update timer if it's running."""
        if self._update_timer is not None:
            self._update_timer.cancel()
            self._update_timer = None

    def update_status(self, future: asyncio.Future) -> None:
        """Update the widget based on the future's status.

        This is called automatically when the task completes.

        Parameters
        ----------
        future
            The completed future/task

        """
        self._stop_periodic_updates()

        if future.cancelled():
            self._refresh_display("cancelled")
        elif future.exception():
            exception = future.exception()
            assert isinstance(exception, Exception)
            self._refresh_display("failed", exception)
        else:
            self._refresh_display("done")

    def display(self) -> None:
        """Display the widget in the current cell."""
        if self._widget is not None:
            display(self._widget)

    def attach_task(self, task: asyncio.Task) -> None:
        """Attach the widget to a task for monitoring.

        Parameters
        ----------
        task
            The asyncio.Task to monitor

        """
        self._task = task

        # Register for completion notification
        task.add_done_callback(self.update_status)

        # Initial display update
        self._refresh_display("running")

        # Start periodic updates
        self._start_periodic_updates()

    def _start_periodic_updates(self) -> None:
        """Start the periodic update timer."""
        try:
            loop = asyncio.get_event_loop()
            self._update_timer = loop.create_task(self._update_periodically())
        except RuntimeError:
            # Fallback if we can't get the event loop
            print("Could not start periodic updates - event loop not available")
        except Exception as e:  # noqa: BLE001
            print(f"Error starting periodic updates: {e}")
