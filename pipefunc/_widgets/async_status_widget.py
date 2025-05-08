from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

import ipywidgets
from IPython.display import HTML, display

if TYPE_CHECKING:
    import asyncio


class AsyncMapStatusWidget:
    """Manages an ipywidgets.Output widget to display the status of an asyncio.Task."""

    def __init__(self) -> None:
        self._widget: ipywidgets.Output | None = None
        self._widget = ipywidgets.Output()
        self._start_time = time.time()
        self._start_datetime = datetime.now().strftime("%H:%M:%S")

        with self._widget:
            self._widget.clear_output(wait=True)
            self._display_status("initializing")

    def _display_status(self, status: str, error: Exception | None = None) -> None:
        """Display a compact, single-line formatted status message."""
        elapsed = time.time() - self._start_time

        # Define styles for different statuses
        styles = {
            "initializing": {
                "color": "#3498db",  # Blue
                "icon": "⏳",
                "message": "Task is initializing...",
            },
            "running": {
                "color": "#3498db",  # Blue
                "icon": "⚙️",
                "message": "Task is running...",
            },
            "done": {
                "color": "#2ecc71",  # Green
                "icon": "✅",
                "message": "Task completed successfully",
            },
            "cancelled": {
                "color": "#f39c12",  # Orange
                "icon": "⚠️",
                "message": "Task was cancelled",
            },
            "failed": {
                "color": "#e74c3c",  # Red
                "icon": "❌",
                "message": "Task failed",
            },
        }

        style = styles.get(status, styles["running"])

        # Build the compact HTML content on a single line
        html_content = f"""
        <div style="display: flex; align-items: center; font-size: 14px; padding: 5px 10px; border-radius: 4px; background-color: #f8f9fa; border-left: 3px solid {style["color"]};">
            <span style="color: {style["color"]}; font-weight: bold; margin-right: 5px;">
                {style["icon"]} {style["message"]}
            </span>
            <span style="color: #666; margin-left: auto; font-size: 12px;">
                Started: {self._start_datetime} | Elapsed: {elapsed:.1f}s
            </span>
        """

        # Add error details if available in a more compact way
        if status == "failed" and error is not None:
            error_text = str(error)
            # Truncate error if too long
            if len(error_text) > 50:
                error_text = error_text[:47] + "..."

            html_content += f"""
            <span style="margin-left: 10px; color: #e74c3c; font-size: 12px;">
                {type(error).__name__}: {error_text}
            </span>
            """

        html_content += "</div>"
        display(HTML(html_content))

    def update_status(self, future: asyncio.Future) -> None:
        """Updates the widget based on the future's status."""
        if self._widget is None:
            return

        with self._widget:
            self._widget.clear_output(wait=True)

            if future.cancelled():
                self._display_status("cancelled")
            elif future.exception():
                self._display_status("failed", future.exception())
            elif not future.done():
                self._display_status("running")
            else:
                self._display_status("done")

    def display(self) -> None:
        """Displays the widget if it exists."""
        if self._widget is not None:
            display(self._widget)

    def attach_task(self, task: asyncio.Task) -> None:
        """Attaches the widget's update logic to the task's completion."""
        if self._widget is not None:
            task.add_done_callback(self.update_status)
