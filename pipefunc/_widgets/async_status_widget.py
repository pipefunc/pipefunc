from __future__ import annotations

from typing import TYPE_CHECKING

import ipywidgets
from IPython.display import display

if TYPE_CHECKING:
    import asyncio


class AsyncMapStatusWidget:
    """Manages an ipywidgets.Output widget to display the status of an asyncio.Task."""

    def __init__(self) -> None:
        self._widget: ipywidgets.Output | None = None
        self._widget = ipywidgets.Output()
        with self._widget:
            self._widget.clear_output(wait=True)
            print("Task is initializing...")

    def update_status(self, future: asyncio.Future) -> None:
        """Updates the widget based on the future's status."""
        if self._widget is None:
            return

        with self._widget:
            self._widget.clear_output(wait=True)
            if future.cancelled():
                print("❌ Task was cancelled.")
            elif future.exception():
                print(f"❌ Task failed: {type(future.exception()).__name__} - {future.exception()}")
            elif not future.done():
                print("⌛ Task is running")
            else:
                print("✅ Task is Done.")

    def display(self) -> None:
        """Displays the widget if it exists."""
        if self._widget is not None:
            display(self._widget)

    def attach_task(self, task: asyncio.Task) -> None:
        """Attaches the widget's update logic to the task's completion."""
        if self._widget is not None:
            task.add_done_callback(self.update_status)
