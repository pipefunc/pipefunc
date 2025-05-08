from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import IPython.display
import ipywidgets
from rich.console import Console
from rich.traceback import Traceback

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

    def __init__(self, initial_update_interval: float = 0.1, *, display: bool = True) -> None:
        """Initialize the status widget."""
        self._main_widget = ipywidgets.VBox()
        self._status_widget = ipywidgets.Output()
        self._traceback_widget = ipywidgets.Output()
        self._traceback_button = None
        self._traceback_visible = False

        self._start_time = time.time()
        self._start_datetime = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
        self._update_interval = initial_update_interval
        self._task: asyncio.Task | None = None
        self._update_timer: asyncio.Task | None = None
        self._exception: Exception | None = None

        self._refresh_display("initializing")
        if display:
            self.display()

    def _get_elapsed_time(self) -> float:
        """Get elapsed time in seconds since widget creation."""
        return time.time() - self._start_time

    def _get_style(self, status: StatusType) -> StyleInfo:
        """Get style information for the given status."""
        return _STYLES.get(status, _STYLES["running"])

    def _toggle_traceback(self, _) -> None:
        """Toggle the visibility of the traceback widget."""
        self._traceback_visible = not self._traceback_visible

        if self._traceback_visible:
            self._traceback_button.description = "Hide traceback"
            self._main_widget.children = (
                self._status_widget,
                self._traceback_button,
                self._traceback_widget,
            )
        else:
            self._traceback_button.description = "Show traceback"
            self._main_widget.children = (self._status_widget, self._traceback_button)

    def _refresh_display(self, status: StatusType, error: Exception | None = None) -> None:
        """Refresh the display with current status."""
        with self._status_widget:
            self._status_widget.clear_output(wait=True)
            style = self._get_style(status)
            elapsed = self._get_elapsed_time()

            # Create container with nice styling
            container_style = "; ".join(
                [
                    "display: flex",
                    "flex-direction: column",
                    "font-size: 13px",
                    "padding: 5px 8px",
                    "border-radius: 4px",
                    "background-color: #f8f9fa",
                    f"border-left: 3px solid {style.color}",
                    "margin-bottom: 5px",
                ],
            )

            # Build status line with better styling
            status_html = f"""
            <div style="{container_style}">
                <div style="display: flex; align-items: center; width: 100%">
                    <span style="color: {style.color}; font-weight: bold; margin-right: 10px;">
                        {style.icon} {style.message}
                    </span>
                    <span style="color: #666; margin-left: auto; font-size: 12px;">
                        Started: {self._start_datetime} | Elapsed: {elapsed:.1f}s
                    </span>
                </div>
            """

            # Add error details with better styling
            if status == "failed" and error is not None:
                status_html += f"""
                <div style="color: #e74c3c; font-weight: bold; margin-top: 8px; font-size: 14px;">
                    {type(error).__name__}: {error!s}
                </div>
                """

            status_html += "</div>"
            IPython.display.display(IPython.display.HTML(status_html))

        # Update the traceback widgets if there's an error
        if status == "failed" and error is not None:
            self._exception = error

            # Create a styled button
            if self._traceback_button is None:
                self._traceback_button = ipywidgets.Button(
                    description="Show traceback",
                    button_style="info",
                    layout=ipywidgets.Layout(width="100%"),
                )
                self._traceback_button.add_class("custom-button")
                self._traceback_button.on_click(self._toggle_traceback)

            # Update the traceback with rich HTML output
            with self._traceback_widget:
                self._traceback_widget.clear_output(wait=True)

                # Create a string IO console with HTML output
                console = Console(
                    width=100,
                    color_system="truecolor",
                    record=True,
                    highlight=True,
                    force_jupyter=True,
                )

                tb = Traceback.from_exception(
                    type(error),
                    error,
                    error.__traceback__,
                    width=100,
                    show_locals=False,
                )

                # # Print and capture the rich traceback
                console.print(tb)

            # Update the main widget structure
            self._main_widget.children = (self._status_widget, self._traceback_button)
        else:
            # No error, just show the status widget
            self._main_widget.children = (self._status_widget,)

    async def _update_periodically(self) -> None:
        """Periodically update the widget while the task is running."""
        if self._task is None:
            return

        try:
            while not self._task.done():
                self._refresh_display("running")
                await asyncio.sleep(self._update_interval)
                elapsed = self._get_elapsed_time()
                if elapsed < 10:  # noqa: PLR2004
                    pass
                elif elapsed < 100:  # noqa: PLR2004
                    self._update_interval = 1.0
                elif elapsed < 1000:  # noqa: PLR2004
                    self._update_interval = 10.0
                else:
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
        IPython.display.display(self._main_widget)

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
